import streamlit as st
import numpy as np
import joblib
import os # 파일 경로 확인을 위해 추가

st.title("신체 정보를 이용한 몸무게 예측 머신러닝 모델")
st.write("신체 정보를 입력하면 몸무게를 예측합니다.")

st.sidebar.header("머신러닝 모델 설계 실습 (다중회귀)")

# [해결책] 현재 파이썬 파일이 있는 위치를 기준으로 절대 경로를 잡습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "weight_model_male.pkl")

# 스트림릿 성능 최적화 및 안정성을 위해 캐싱 함수 활용
@st.cache_resource
def load_my_model(path):
    return joblib.load(path)

# 모델 로드 시도 및 예외 처리
try:
    model = load_my_model(model_path)
    model_loaded = True
except FileNotFoundError:
    st.error(f"🚨 모델 파일을 찾을 수 없습니다! \n현재 경로에 파일이 있는지 확인해주세요: `{model_path}`")
    model_loaded = False
except Exception as e:
    st.error(f"🚨 모델 로드 중 오류가 발생했습니다: {e}")
    st.warning("💡 모델을 저장할 때의 scikit-learn 버전과 현재 버전이 맞지 않을 수 있습니다.")
    model_loaded = False

# 입력 UI
height = st.slider("키 (cm)", 140.0, 190.0, 170.0)
waist = st.slider("허리 둘레 (cm)", 50.0, 120.0, 80.0)
hip = st.slider("엉덩이 둘레 (cm)", 85.0, 120.0, 100.0)

X = np.array([[height, waist, hip]])

if st.button("몸무게 예측하기"):
    if model_loaded:
        prediction = model.predict(X)
        st.success(f"예측 몸무게 : {prediction[0]:.1f} kg")
    else:
        st.error("모델이 로드되지 않아 예측할 수 없습니다.")
