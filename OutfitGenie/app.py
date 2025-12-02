import streamlit as st
import pandas as pd
import datetime
import requests
import base64
import json
from pathlib import Path
from openai import OpenAI


# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(page_title="OutfitGenie – AI 코디네이터", layout="wide")
client = OpenAI()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

WARDROBE_FILE = DATA_DIR / "wardrobe.csv"


# ============================================================
# Base64 이미지 저장/로드
# ============================================================

def encode_image_to_base64(file_bytes: bytes) -> str:
    """이미지 파일을 Base64 문자열로 변환."""
    return base64.b64encode(file_bytes).decode("utf-8")


def decode_base64_to_image(b64_string: str):
    """Base64 문자열을 이미지 바이트로 디코딩."""
    try:
        return base64.b64decode(b64_string)
    except:
        return None


# ============================================================
# 옷장 데이터 로드/저장
# ============================================================

def load_wardrobe():
    if not WARDROBE_FILE.exists():
        df = pd.DataFrame(
            columns=[
                "id", "type", "name", "color",
                "pattern", "formality", "season",
                "thickness", "image_base64"
            ]
        )
        df.to_csv(WARDROBE_FILE, index=False, encoding="utf-8-sig")
        return df

    df = pd.read_csv(WARDROBE_FILE, encoding="utf-8-sig")

    required_cols = [
        "id", "type", "name", "color", "pattern",
        "formality", "season", "thickness", "image_base64"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    return df


def save_wardrobe(df: pd.DataFrame):
    df.to_csv(WARDROBE_FILE, index=False, encoding="utf-8-sig")


def generate_item_id(df: pd.DataFrame):
    if df.empty:
        return "item_1"

    nums = (
        df["id"].astype(str)
        .str.replace("item_", "", regex=False)
        .fillna("0")
        .astype(int)
    )
    return f"item_{nums.max() + 1}"


# ============================================================
# AI Vision - 사진 분석하여 메타데이터 생성
# ============================================================

def analyze_image_with_ai(image_bytes: bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    system_prompt = """
당신은 패션 이미지 분석 전문 AI입니다.
입력 이미지를 보고 옷의 종류, 색상, 계절감 등을 JSON으로 반환하세요.

JSON 형식 예:
{
  "type_ko": "상의 | 하의 | 아우터",
  "name_suggestion": "옷을 잘 표현한 한국어 이름",
  "color_main_ko": "주요 색상",
  "color_sub_ko": "보조 색상 (없으면 null)",
  "pattern_ko": "무지 | 스트라이프 | 체크 | 기타",
  "formality_ko": "캐주얼 | 세미 정장 | 정장",
  "season_ko": ["봄", "여름"],
  "thickness": 1
}

설명 없이 JSON만 반환하세요.
"""

    user_prompt = "이 옷 사진을 분석하여 JSON을 반환해 주세요."

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }
                ]
            }
        ]
    )

    raw = res.choices[0].message.content.strip()
    raw = raw.strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()

    try:
        return json.loads(raw)
    except:
        return None


# ============================================================
# 기상청 API (Base)
# ============================================================

KMA_API_KEY = st.secrets.get("KMA_API_KEY")

CITY_GRID = {
    ("서울특별시", "강남구"): (61, 125),
    ("서울특별시", "강북구"): (61, 130),
    ("서울특별시", "강서구"): (58, 126),
    ("경기도", "수원시"): (60, 121),
}


def get_kma_weather(nx, ny):
    if not KMA_API_KEY:
        return None

    now = datetime.datetime.now()
    base_date = now.strftime("%Y%m%d")
    base_hour = (now.hour - 1) // 3 * 3 + 2
    if base_hour < 2:
        base_date = (now - datetime.timedelta(days=1)).strftime("%Y%m%d")
        base_hour = 23

    base_time = f"{base_hour:02d}00"

    url = (
        "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        f"?serviceKey={KMA_API_KEY}&numOfRows=300&pageNo=1&dataType=JSON"
        f"&base_date={base_date}&base_time={base_time}&nx={nx}&ny={ny}"
    )

    try:
        res = requests.get(url, timeout=10).json()
        items = res["response"]["body"]["items"]["item"]

        sky, temp = None, None
        for it in items:
            if it["category"] == "SKY":
                sky = it["fcstValue"]
            elif it["category"] == "T1H":
                temp = float(it["fcstValue"])

    except:
        return None

    if sky == "1":
        sky = "맑음"
    elif sky == "3":
        sky = "구름 많음"
    elif sky == "4":
        sky = "흐림"
    else:
        sky = "알 수 없음"

    return {
        "sky": sky,
        "temp": temp,
        "temp_text": f"{temp:.1f}℃" if temp else "?"
    }


# ============================================================
# AI 코디 추천
# ============================================================

def ai_coordinate(wardrobe_text, weather, purpose, time, province, district):
    system_prompt = """
당신은 전문 패션 코디네이터입니다.
오직 옷장에 존재하는 옷만 조합하며,
날씨·목적·시간대 기반으로 최적의 코디를 추천합니다.

반드시 한국어로, 아래 형식을 지켜 출력하세요:

[추천 코디]
상의: …
하의: …

[선택 이유]
상의: …
하의: …

[한 줄 요약]
…
"""

    user_prompt = f"""
[옷장 목록]
{wardrobe_text}

[상황 정보]
지역: {province} {district}
날씨: {weather['sky']}
기온: {weather['temp_text']}
목적: {purpose}
시간대: {time}

위 조건에 맞게 가장 어울리는 상·하의 조합을 추천하세요.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return res.choices[0].message.content


# ============================================================
# UI 시작
# ============================================================

st.title("🧥 OutfitGenie – AI 코디네이터")
menu = st.sidebar.radio("메뉴 선택", ["옷 등록", "옷장 보기", "AI 코디 추천"])

wardrobe = load_wardrobe()


# ============================================================
# 1. 옷 등록
# ============================================================

if menu == "옷 등록":
    st.header("📸 옷 사진 업로드")

    uploaded = st.file_uploader("옷 사진 선택", type=["jpg", "jpeg", "png"])

    if uploaded:
        img_bytes = uploaded.getvalue()
        st.image(img_bytes, caption="미리보기", use_column_width=True)

        if st.button("🧠 AI 자동 분석"):
            with st.spinner("AI가 분석 중입니다..."):
                info = analyze_image_with_ai(img_bytes)

            if info:
                st.success("AI 분석 완료!")
                st.write(info)
                st.session_state["auto_info"] = info
            else:
                st.error("AI 분석 실패")

        auto = st.session_state.get("auto_info", {})

        # 자동 입력 기본값
        type_default = auto.get("type_ko", "상의")
        name_default = auto.get("name_suggestion", "")
        color_default = auto.get("color_main_ko", "")

        pattern_default = auto.get("pattern_ko", "무지")
        formality_default = auto.get("formality_ko", "캐주얼")
        season_default = auto.get("season_ko", [])
        thickness_default = auto.get("thickness", 3)

        type_map = {"상의": "top", "하의": "bottom", "아우터": "outer"}

        item_type = st.selectbox("종류", ["상의", "하의", "아우터"], index=["상의", "하의", "아우터"].index(type_default))
        name = st.text_input("이름", value=name_default)
        color = st.text_input("주요 색상", value=color_default)

        pattern = st.selectbox("패턴", ["무지", "스트라이프", "체크", "기타"], index=["무지", "스트라이프", "체크", "기타"].index(pattern_default))
        formality = st.selectbox("격식", ["캐주얼", "세미 정장", "정장"], index=["캐주얼", "세미 정장", "정장"].index(formality_default))
        season = st.multiselect("계절", ["봄", "여름", "가을", "겨울"], default=season_default)
        thickness = st.slider("두께", 1, 5, int(thickness_default))

        if st.button("저장하기"):
            b64 = encode_image_to_base64(img_bytes)
            item_id = generate_item_id(wardrobe)

            new_row = {
                "id": item_id,
                "type": type_map[item_type],
                "name": name,
                "color": color,
                "pattern": pattern,
                "formality": formality,
                "season": ",".join(season),
                "thickness": thickness,
                "image_base64": b64,
            }

            wardrobe = pd.concat([wardrobe, pd.DataFrame([new_row])], ignore_index=True)
            save_wardrobe(wardrobe)
            st.success("저장 완료!")


# ============================================================
# 2. 옷장 보기
# ============================================================

elif menu == "옷장 보기":
    st.header("👚 내 옷장")

    if wardrobe.empty:
        st.info("저장된 옷이 없습니다.")
    else:
        cols = st.columns(3)
        for idx, row in wardrobe.iterrows():
            with cols[idx % 3]:
                st.markdown("---")
                img_b = decode_base64_to_image(row["image_base64"])
                if img_b:
                    st.image(img_b, width=260, caption=row["name"])
                else:
                    st.warning("이미지 오류")

                if st.button("삭제", key=f"del_{idx}"):
                    wardrobe = wardrobe.drop(idx)
                    save_wardrobe(wardrobe)
                    st.rerun()


# ============================================================
# 3. AI 코디 추천
# ============================================================

elif menu == "AI 코디 추천":
    st.header("🤖 AI 코디 추천")

    province = st.selectbox("도/특별시", ["서울특별시", "경기도"])
    district = st.selectbox("구/시", ["강남구", "강북구", "강서구"] if province == "서울특별시" else ["수원시"])

    grid = CITY_GRID.get((province, district))
    weather = get_kma_weather(*grid) if grid else None

    if weather:
        st.success(f"자동 날씨: {weather['sky']} / {weather['temp_text']}")
    else:
        st.info("자동 불러오기 실패. 수동 입력 사용.")
        sky = st.selectbox("하늘 상태", ["맑음", "구름 많음", "흐림", "비", "눈"])
        temp = st.number_input("기온 (℃)", value=20)
        weather = {"sky": sky, "temp": temp, "temp_text": f"{temp}℃"}

    purpose = st.selectbox("오늘의 목적", ["출근/통학", "친구 만남", "데이트", "면접", "가벼운 산책"])
    time = st.selectbox("시간대", ["아침", "낮", "저녁", "밤"])

    if st.button("AI 코디 추천 받기"):
        if wardrobe.empty:
            st.warning("옷장이 비어 있습니다.")
        else:
            wardrobe_text = "\n".join([
                f"- {row['id']} / 종류:{row['type']} / 이름:{row['name']} / 색상:{row['color']}"
                for _, row in wardrobe.iterrows()
            ])

            with st.spinner("AI가 코디 중..."):
                result = ai_coordinate(wardrobe_text, weather, purpose, time, province, district)

            st.markdown("### 🧾 추천 결과")
            st.markdown(result)
