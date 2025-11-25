import streamlit as st
import pandas as pd
import os
import datetime
import requests
from pathlib import Path
from openai import OpenAI

# ===================== 기본 설정 =====================
st.set_page_config(page_title="OutfitGenie – AI 코디네이터", layout="wide")

# OpenAI 클라이언트 (환경변수에 OPENAI_API_KEY가 있어야 함)
client = OpenAI()

# 데이터/이미지 폴더
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "images"
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

wardrobe_path = DATA_DIR / "wardrobe.csv"

# ===================== 유틸 함수 =====================

def load_wardrobe():
    """옷장 CSV 로드 (없으면 기본 형태로 생성)"""
    if not wardrobe_path.exists():
        df = pd.DataFrame(
            columns=[
                "id",
                "type",       # top / bottom / outer
                "name",       # 옷 이름
                "color",      # 주요 색상
                "pattern",    # 무늬
                "formality",  # 격식 (캐주얼/포멀 등)
                "season",     # 계절
                "thickness",  # 두께 (1~5)
                "image_path", # 이미지 파일 경로
            ]
        )
        df.to_csv(wardrobe_path, index=False, encoding="utf-8-sig")
    else:
        df = pd.read_csv(wardrobe_path, encoding="utf-8-sig")
    # 결측 컬럼 보정
    for col in [
        "id",
        "type",
        "name",
        "color",
        "pattern",
        "formality",
        "season",
        "thickness",
        "image_path",
    ]:
        if col not in df.columns:
            df[col] = ""
    return df


def save_wardrobe(df: pd.DataFrame):
    df.to_csv(wardrobe_path, index=False, encoding="utf-8-sig")


def generate_item_id(df: pd.DataFrame) -> str:
    """item_n 형태의 ID 생성"""
    if df.empty:
        return "item_1"
    # 숫자 부분만 뽑아서 +1
    nums = (
        df["id"]
        .astype(str)
        .str.replace("item_", "", regex=False)
        .fillna("0")
        .astype(int)
    )
    return f"item_{nums.max() + 1}"


# ===================== 기상청 관련 (자동 모드) =====================

# 간단한 격자 좌표 샘플 (실제론 더 많이 넣어도 됨)
CITY_GRID = {
    ("서울특별시", "강남구"): (61, 125),
    ("서울특별시", "강북구"): (61, 130),
    ("서울특별시", "강서구"): (58, 126),
    ("경기도", "수원시"): (60, 121),
}

KMA_API_KEY = st.secrets.get("KMA_API_KEY", None)


def get_kma_weather(nx: int, ny: int):
    """기상청 단기예보를 통해 하늘 상태/기온 등을 가져오는 예시 함수
       - 실패하면 (None, 에러메시지) 반환
    """
    if not KMA_API_KEY:
        return None, "기상청 API 키가 설정되어 있지 않습니다."

    # 기준 시각 계산 (기상청 단기예보는 1~3시간 단위)
    now = datetime.datetime.now()
    base_date = now.strftime("%Y%m%d")

    # 02, 05, 08 ... 식으로 가장 최근 발표 시각 찾기
    base_hour = (now.hour - 1) // 3 * 3 + 2
    if base_hour < 2:
        # 새벽 0~1시는 전날 23시 발표 사용 등
        base_hour = 23
        base_date = (now - datetime.timedelta(days=1)).strftime("%Y%m%d")
    base_time = f"{base_hour:02d}00"

    url = (
        "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        f"?serviceKey={KMA_API_KEY}"
        f"&numOfRows=1000&pageNo=1&dataType=JSON"
        f"&base_date={base_date}&base_time={base_time}"
        f"&nx={nx}&ny={ny}"
    )

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        items = data["response"]["body"]["items"]["item"]
    except Exception as e:
        return None, f"기상청 API 호출 실패: {e}"

    # 오늘(또는 현재시각 이후) 가장 가까운 시간대 하나만 뽑는 간단 로직
    sky = None
    t1h = None

    for it in items:
        category = it["category"]
        fcst_value = it["fcstValue"]
        if category == "SKY":
            # 1:맑음, 3:구름많음, 4:흐림
            sky = fcst_value
        elif category == "T1H":
            t1h = float(fcst_value)

    if sky is None and t1h is None:
        return None, "기상 데이터를 찾지 못했습니다."

    # 사람이 읽기 좋은 텍스트로 변환
    if sky == "1":
        sky_text = "맑음"
    elif sky == "3":
        sky_text = "구름 많음"
    elif sky == "4":
        sky_text = "흐림"
    else:
        sky_text = "알 수 없음"

    temp_text = f"{t1h:.1f}℃" if t1h is not None else "알 수 없음"

    weather = {
        "sky": sky_text,
        "temp": t1h,
        "temp_text": temp_text,
    }
    return weather, None


def manual_weather_input():
    """사용자가 직접 날씨를 입력하는 폼"""
    st.info("기상청 데이터를 불러올 수 없습니다. 직접 입력해 주세요.")
    sky = st.selectbox("하늘 상태", ["맑음", "구름 많음", "흐림", "비", "눈"], index=0)
    temp = st.number_input("현재 기온 (℃)", value=20, step=1)
    # 체감기온 등은 생략
    return {
        "sky": sky,
        "temp": temp,
        "temp_text": f"{temp:.1f}℃",
    }


# ===================== AI 코디네이터 프롬프트 =====================

def ai_coordinate(wardrobe_text: str, weather: dict, purpose: str, time: str, province: str, district: str) -> str:
    """OpenAI GPT로 코디 추천"""
    sky = weather.get("sky", "알 수 없음")
    temp_text = weather.get("temp_text", "알 수 없음")

    system_prompt = """
당신은 사용자의 옷장 정보를 보고 오늘의 날씨, 목적, 시간대를 고려해 최적의 상·하의 코디를 추천하는 패션 코디네이터 AI입니다.
규칙:
- 반응은 반드시 한국어로 합니다.
- 상의 1벌, 하의 1벌을 반드시 선택합니다. (outer는 선택할 수 있으면 참고 정도만)
- 각 아이템은 옷장 목록에 실제 존재하는 이름으로만 선택합니다.
- 추천하는 상의/하의가 각각 어떤 이유로 선택되었는지 자세하게 설명합니다.
- 마지막 줄에 한 줄 요약 코멘트를 넣어 줍니다.
"""

    user_prompt = f"""
[옷장 목록]
{wardrobe_text}

[오늘 정보]
- 지역: {province} {district}
- 하늘 상태: {sky}
- 기온: {temp_text}
- 목적: {purpose}
- 시간대: {time}

위 정보를 바탕으로 오늘 입기 좋은 코디를 추천해 주세요.

형식 예시는 다음과 같이 해 주세요.

[추천 코디]
- 상의: 상의1 이름
- 하의: 하의2 이름

[선택 이유]
- 상의: ...
- 하의: ...

[한 줄 요약]
...
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.7,
    )

    return res.choices[0].message.content


# ===================== UI 시작 =====================

st.title("🧥 OutfitGenie — AI 코디네이터")

menu = st.sidebar.radio("메뉴 선택", ["옷 등록", "옷장 보기", "AI 코디 추천"])

wardrobe = load_wardrobe()

# ---------------------------------------------------
# 1. 옷 등록
# ---------------------------------------------------
if menu == "옷 등록":
    st.header("📸 옷 사진 업로드")

    uploaded = st.file_uploader(
        "옷 사진 선택",
        type=["png", "jpg", "jpeg"],
        help="실제 입고 있는 옷 사진을 업로드해 주세요.",
    )

    if uploaded:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded, caption="미리보기", use_column_width=True)

        with col2:
            st.subheader("옷 정보 입력")

            item_type = st.selectbox("종류", ["상의", "하의", "아우터"])
            type_map = {"상의": "top", "하의": "bottom", "아우터": "outer"}

            name = st.text_input("옷 이름 (예: 아이보리 니트, 검정 슬랙스)")
            color = st.text_input("주요 색상 (예: 아이보리, 검정)")
            pattern = st.selectbox("패턴", ["솔리드(무늬 없음)", "스트라이프", "체크", "기타"])
            formality = st.selectbox("분위기 / 격식", ["캐주얼", "세미 정장", "정장"])
            season = st.multiselect("계절", ["봄", "여름", "가을", "겨울"], max_selections=4)
            thickness = st.slider("두께 (1=매우 얇음, 5=매우 두꺼움)", 1, 5, 3)

            if st.button("옷장에 저장하기"):
                if not uploaded:
                    st.error("사진이 있어야 저장할 수 있습니다.")
                else:
                    # 이미지 저장
                    ext = uploaded.name.split(".")[-1]
                    item_id = generate_item_id(wardrobe)
                    filename = f"{item_id}.{ext}"
                    save_path = IMAGE_DIR / filename
                    with open(save_path, "wb") as f:
                        f.write(uploaded.getbuffer())

                    # 데이터프레임에 추가
                    new_row = {
                        "id": item_id,
                        "type": type_map[item_type],
                        "name": name,
                        "color": color,
                        "pattern": pattern,
                        "formality": formality,
                        "season": ",".join(season),
                        "thickness": thickness,
                        "image_path": str(save_path),
                    }
                    new_row_df = pd.DataFrame([new_row])
                    wardrobe = pd.concat([wardrobe, new_row_df], ignore_index=True)
                    save_wardrobe(wardrobe)

                    st.success("옷장이 저장되었습니다!")

# ---------------------------------------------------
# 2. 옷장 보기
# ---------------------------------------------------
elif menu == "옷장 보기":
    st.header("👚 내 옷장")

    if wardrobe.empty:
        st.info("아직 등록된 옷이 없습니다. 먼저 '옷 등록'에서 옷을 추가해 주세요.")
    else:
        # 아이콘/그리드 형식으로 보여주기
        st.write("### 등록된 옷 목록")

        # 3열 그리드
        cols = st.columns(3)

        for idx, row in wardrobe.iterrows():
            col = cols[idx % 3]
            with col:
                st.markdown("---")

                # 이미지 표시 (파일이 실제로 존재할 때만)
                img_path = row["image_path"]
                if isinstance(img_path, str) and img_path and os.path.exists(img_path):
                    st.image(
                        img_path,
                        width=250,
                        caption=row["name"] if row["name"] != "" else "(이름 없음)",
                    )
                else:
                    st.warning("이미지 없음 (파일을 찾을 수 없습니다)")

                # 삭제 버튼
                if st.button("🗑 삭제", key=f"del_{idx}"):
                    wardrobe = wardrobe.drop(idx)
                    save_wardrobe(wardrobe)
                    st.success("삭제되었습니다!")
                    st.rerun()

# ---------------------------------------------------
# 3. AI 코디 추천
# ---------------------------------------------------
elif menu == "AI 코디 추천":
    st.header("🤖 AI 코디 추천")

    # ------------ 지역 선택 ------------
    st.subheader("1️⃣ 지역 선택")

    province = st.selectbox("도/특별시 선택", ["서울특별시", "경기도"])
    if province == "서울특별시":
        district = st.selectbox("시/군/구 선택", ["강남구", "강북구", "강서구"])
    elif province == "경기도":
        district = st.selectbox("시/군/구 선택", ["수원시"])
    else:
        district = st.selectbox("시/군/구 선택", ["강남구"])

    # 격자 좌표 조회
    grid = CITY_GRID.get((province, district), None)

    # ------------ 날씨 자동/수동 결정 ------------
    weather = None
    error_msg = None

    if grid is not None and KMA_API_KEY:
        nx, ny = grid
        weather, error_msg = get_kma_weather(nx, ny)

    if weather is None:
        # 자동 실패 → 수동 입력
        weather = manual_weather_input()
    else:
        st.success(
            f"기상청 자동 불러오기 성공: 하늘 상태 {weather['sky']}, 기온 {weather['temp_text']}"
        )
        # 원하면 사용자가 수정할 수 있도록 간단한 편집도 허용
        with st.expander("기상 정보를 직접 수정하고 싶다면 펼쳐서 조정하세요.", expanded=False):
            sky = st.selectbox(
                "하늘 상태 (수정 가능)",
                ["맑음", "구름 많음", "흐림", "비", "눈"],
                index=["맑음", "구름 많음", "흐림", "비", "눈"].index(weather["sky"])
                if weather.get("sky") in ["맑음", "구름 많음", "흐림", "비", "눈"]
                else 0,
            )
            temp = st.number_input(
                "현재 기온 (℃, 수정 가능)",
                value=float(weather["temp"]) if weather.get("temp") is not None else 20.0,
                step=1.0,
            )
            weather["sky"] = sky
            weather["temp"] = temp
            weather["temp_text"] = f"{temp:.1f}℃"

    # ------------ 목적 / 시간대 ------------
    st.subheader("2️⃣ 오늘의 상황")

    purpose = st.selectbox("오늘의 목적", ["출근/통학", "친구 만남", "데이트", "면접", "가벼운 산책"])
    time = st.selectbox("시간대", ["아침", "낮", "저녁", "밤"])

    # ------------ AI 호출 ------------
    if wardrobe.empty:
        st.warning("옷장에 등록된 옷이 없어서 AI가 코디를 추천할 수 없습니다.")
    else:
        if st.button("AI 코디 추천 받기"):
            # 옷장 텍스트 정리
            lines = []
            for _, row in wardrobe.iterrows():
                lines.append(
                    f"- {row['id']} / 종류:{row['type']} / 이름:{row['name']} / 색상:{row['color']} / 계절:{row['season']} / 두께:{row['thickness']}"
                )
            wardrobe_text = "\n".join(lines)

            with st.spinner("AI가 코디를 고민 중입니다..."):
                result = ai_coordinate(wardrobe_text, weather, purpose, time, province, district)

            st.markdown("### 🧾 AI 추천 결과")
            st.markdown(result)
