import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 시스템에 따라 폰트 경로 다를 수 있음 (NanumGothic 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(font='AppleGothic')


# 데이터 호출
## 다른 유저 데이터
encoded_df = pd.read_csv('./data/users_encoded_df.csv', encoding='utf-8-sig')


## 내 데이터
encoded_2_df = pd.read_csv('./data/my_encoded_df.csv', encoding='utf-8-sig')


# 스루패스 관련 임시 데이터 프레임 가공
throughPass_df = encoded_df.groupby(['matchId', 'ouid', 'matchResult'])[['throughPassTry', 'throughPassSuccess']].sum().reset_index()
throughPass_tmp_df = throughPass_df.groupby('matchResult')[['throughPassTry','throughPassSuccess']].mean().reset_index()
throughPass_2_df = encoded_2_df.groupby(['matchId', 'ouid', 'matchResult'])[['throughPassTry', 'throughPassSuccess']].sum().reset_index()
throughPass_tmp_2_df = throughPass_2_df.groupby('matchResult')[['throughPassTry','throughPassSuccess']].mean().reset_index()



# --- 페이지 구성 설정 ---
st.set_page_config(page_title="FC온라인 분석 대시보드", layout="wide")

# --- 사이드바 ---
st.sidebar.title("📁 페이지 이동")
page = st.sidebar.radio("이동할 페이지를 선택하세요:", ["🏠 프로젝트 소개", "📊 모델 실험", "⚽ 주요 변수 비교 분석"])

# --- 공통 스타일 ---
sns.set_theme(style='whitegrid', font='AppleGothic')

# --- 1. 프로젝트 소개 ---
if page == "🏠 프로젝트 소개":
    
    st.title("🏠 프로젝트 소개")
    st.image("https://www.youthdaily.co.kr/data/photos/20230938/art_16952959138868_7c67ad.jpg", width=800)

    st.markdown("## 프로젝트 배경 및 목표")
    with st.expander("프로젝트 배경", expanded=True):
        st.markdown("""
        - FC온라인 20년차, 아직도 **<프로페셔널>** 티어에 머물고 있고 실력만으로는 극복할 수 없는 벽을 마주함  
        - **승률 29.03%**, 승격/강등을 결정하는 10경기 중 **3경기밖에 승리하지 못하는 상황** → 강등 반복
        """)

    with st.expander("프로젝트 목표", expanded=True):
        st.markdown("""
        1. **승리에 영향을 주는 주요 플레이 특성**을 파악하고 **승리 유저와 나의 플레이를 비교 분석**
        2. 단순한 승패 예측이 아닌, **이기기 위해 어떤 플레이를 강화해야 하는지 인사이트 도출**
        """)

    st.divider()

    st.markdown("## 사용한 스킬 & 도구")
    st.markdown("""
    - **프로그래밍 언어**: Python  
    - **분석 도구**: Pandas, NumPy, Scikit-learn, Statsmodels  
    - **데이터 시각화**: Matplotlib, Seaborn, Streamlit  
    - **ETL 파이프라인**: Airflow  
    - **모델링**: LightGBM, RandomForest, Ensemble, Logistic Regression  
    """)

    st.divider()

    st.markdown("## 활용한 데이터셋")
    st.markdown("""
    - FC온라인 **공식 API**를 통해 수집한 **경기 데이터**
    - **선수 정보 제외**, **플레이 데이터만 활용**
    """)

    st.divider()

    st.markdown("## 프로젝트 진행 과정")
    st.markdown("""
    1. **문제 정의**: 경기 실력만으로 승격이 어려운 상황에서, 어떤 플레이 스타일이 승리에 영향을 미치는지 파악하기 어렵다.
    2. **데이터 수집**: FC온라인 API를 통해 나와 다른 유저 경기 데이터 수집, Airflow로 ETL 파이프라인 자동화  
    3. **데이터 전처리**: 이상치 KNN으로 대치, 비정상종료 매치 제거
    4. **EDA**: 상관관계 분석을 통해 전체 변수간, 승리와 다른 변수, 패배와 다른 변수와의 관계성 파악
    5. **모델링**: 파생 변수 생성, 오버샘플링, 튜닝 방법, 모델 종류 등으로 케이스 구분 후 실험 진행하여 최적 모델 선정
    6. **데이터 분석**: 승리한 유저와 내 플레이 데이터(주요 변수) 비교 분석, 로지스틱 회귀분석으로 수비 지표 정밀 분석
    7. **인사이트 도출 및 실행 계획**: 내 플레이 보완점 반영하여 세부 전술 수정 및 플레이 스타일 개선
    """)

    st.divider()

    st.markdown("## 주요 의사결정")
    st.markdown("""
    - **ETL 자동화** : 반복 실험과 데이터 정합성 확보, 유지관리 편의성 등을 고려해 자동화 구성
    - **이상치 처리** : 이상치가 실제 존재 가능한 범주에 속하고 모수가 적어 제거 시 왜곡 가능성이 높아 KNN으로 대치하여 보존
    - **Optuna 기반 튜닝** : 효율적 탐색 가능, 실험 반복과 시간 효율성 고려
    - **분류 지표로 F1 / ROC AUC 채택** : 클래스 불균형 상황에서 정밀도와 재현율의 균형을 평가하고, 분류 성능을 종합적으로 판단 위해
    - **로지스틱 회귀로 수비 지표 해석** : 단순 상관관계로 설명되지 않는 변수에 대해 수치적 근거를 더하기 위해
    """)

    st.divider()

    st.markdown("## 결과 및 성과")

    with st.expander("결과"):
        st.markdown("""
        - 모델 성능(AUC 0.7148, F1-score 0.673) / 주요 변수(스루패스, 짧은 패스, 유효 슈팅, 롱패스, 블록 수)
        """)

    with st.expander("성과"):
        st.markdown("""
        - 승리에 영향을 주는 플레이 패턴을 정량적으로 분석하고 맞춤형 전략 수립하여 승률 21%p 증가
        - 반복적인 API 호출 및 정상 경기 여부 확인 과정을 Airflow 기반 파이프라인으로 자동화하여, 매일 수동 실행·점검으로 인한 리소스를 절감하고 데이터 수집의 신뢰성과 효율성을 동시에 확보  
        """)

    st.divider()

    st.markdown("## 러닝 포인트")
    st.markdown("""
    - **맥락 기반 분석**의 중요성 체감 → 단순 수치 나열 X  
    - **ETL 자동화**의 비용 절감 효과 실감  
    - 다양한 실험 설계를 통한 **모델 성능 최적화 전략 체득**  
    - **모델 해석 & 인과 해석**을 통한 데이터 기반 개선안 도출 역량 강화  
    """)

# --- 2. 모델 실험 페이지 ---
elif page == "📊 모델 실험":
    st.title("모델 실험 결과 분석")
    
    # 실험 데이터프레임 불러오기
    df = pd.read_csv("./data/fc_online_experiment_results.csv")  # CSV로 저장해두었다고 가정

    # 좌우 컬럼 나누기
    col1, col2 = st.columns([1, 3])  # 비율 조정해서 카드와 표가 예쁘게 나옴
        
    with col1:
        st.markdown("### 🏆 최적 모델 조합")
        st.markdown("""
        - **모델**: 앙상블  
        - **튜닝 방식**: Optuna  
        - **오버샘플링**: ❌ 미적용  
        - **파생변수 생성**: ❌ 미적용  
        - **F1 Score**: **0.5814**  
        - **ROC AUC**: **0.7148**
        """)

    with col2:
        selected_model = st.selectbox("모델 선택", df['모델'].unique())
        use_engineered_features = st.checkbox("파생변수 생성 포함", value=True)

        filtered_df = df[
            (df['모델'] == selected_model) &
            (df['파생변수 생성'] == ("O" if use_engineered_features else "X"))
        ]

        st.markdown(f"#### 🔍 {selected_model} 실험 결과 (파생변수 생성: {'O' if use_engineered_features else 'X'})")
        st.dataframe(filtered_df)

    st.divider()

    # 성능 비교 그래프
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 🎯 F1 Score")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=filtered_df, x='튜닝 방식', y='F1 Score', hue='오버샘플링', palette="Blues_d", ax=ax1)
        ax1.set_title("F1 Score 비교")
        ax1.set_ylim(0.4, 0.7)
        st.pyplot(fig1)

    with col4:
        st.markdown("#### 🎯 ROC AUC")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=filtered_df, x='튜닝 방식', y='ROC AUC', hue='오버샘플링', palette="Greens_d", ax=ax2)
        ax2.set_title("ROC AUC 비교")
        ax2.set_ylim(0.6, 0.75)
        st.pyplot(fig2)


# --- 3. 주요 변수 비교 분석 페이지 ---
elif page == "⚽ 주요 변수 비교 분석":
    st.title("📊 주요 변수 비교 분석")

    selected_var = st.selectbox("비교할 변수를 선택하세요", ["스루패스"])

    if selected_var == "스루패스":
        st.subheader("1. 스루패스 비교")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**📌 다른 유저의 경기 결과별 스루패스 시도 및 성공**")
            fig1, axes1 = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)
            sns.barplot(data=throughPass_tmp_df, x='matchResult', y='throughPassTry', hue='matchResult', ax=axes1[0])
            axes1[0].set_title('스루패스 시도')
            sns.barplot(data=throughPass_tmp_df, x='matchResult', y='throughPassSuccess', hue='matchResult', ax=axes1[1])
            axes1[1].set_title('스루패스 성공')
            sns.despine()
            plt.ylim(10)
            plt.tight_layout()
            st.pyplot(fig1)

            # 평균 스루패스 데이터 계산
            tmp_avg_df = throughPass_df[throughPass_df['matchResult']=='승']
            
            success_rate = round(sum(tmp_avg_df['throughPassSuccess']) / sum(tmp_avg_df['throughPassTry']), 5)
            mean_avg_try = round(tmp_avg_df['throughPassTry'].mean(), 4)
            mean_avg_success = round(tmp_avg_df['throughPassSuccess'].mean(), 4)

            # 표로 보여주기
            st.markdown("**📊 승리 유저들의 평균 스루패스 데이터**")
            avg_df = pd.DataFrame({
                "스루패스 평균 시도 수": [mean_avg_try],
                "스루패스 평균 성공 수": [mean_avg_success],
                "스루패스 성공률": [success_rate]
            })
            st.dataframe(avg_df)
            
        with col2:
            st.markdown("**📌 나의 경기 결과별 스루패스 시도 및 성공**")
            fig2, axes2 = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)
            sns.barplot(data=throughPass_tmp_2_df, x='matchResult', y='throughPassTry', hue='matchResult', ax=axes2[0])
            axes2[0].set_title('스루패스 시도')
            sns.barplot(data=throughPass_tmp_2_df, x='matchResult', y='throughPassSuccess', hue='matchResult', ax=axes2[1])
            axes2[1].set_title('스루패스 성공')
            sns.despine()
            plt.ylim(10)
            plt.tight_layout()
            st.pyplot(fig2)

            # 나의 평균 스루패스 데이터 계산
            success_rate = round(sum(throughPass_2_df['throughPassSuccess']) / sum(throughPass_2_df['throughPassTry']), 5)
            mean_avg_try = round(throughPass_2_df['throughPassTry'].mean(), 4)
            mean_avg_success = round(throughPass_2_df['throughPassSuccess'].mean(), 4)

            # 표로 보여주기
            st.markdown("**📊 나의 평균 스루패스 데이터**")
            avg_df = pd.DataFrame({
                "스루패스 평균 시도 수": [mean_avg_try],
                "스루패스 평균 성공 수": [mean_avg_success],
                "스루패스 성공률": [success_rate]
            })
            st.dataframe(avg_df)

        # 인사이트
        st.markdown("---")
        st.markdown("""
        ### 🧠 인사이트
        - 승리한 유저의 평균 스루패스 시도, 성공, 성공률이 비기고 패한 유저보다 높다.  
        - 스루패스 성공률은 승리한 유저 대비 내가 더 높지만, 평균 시도 횟수가 약 12% 적음  
        - **따라서 스루패스를 더 많이 시도해야 한다.**
        """)