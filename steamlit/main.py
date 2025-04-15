import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import statsmodels.api as sm
import os
import matplotlib.font_manager as fm

# 한글 폰트 설정
# 폰트 파일 경로 (예: 프로젝트 폴더 내 fonts 폴더에 있을 경우)
font_path = "steamlit/fonts/나눔 글꼴/NanumGothic.ttf"

# 폰트 등록
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlib 설정
plt.rc("font", family=font_name)
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# seaborn 설정 (matplotlib 기반이므로 동일 적용)
sns.set_theme(font=font_name)


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

# 짧은패스 관련 임시 데이터 프레임 가공
shortPass_df = encoded_df.groupby(['matchId', 'ouid', 'matchResult'])[['shortPassTry', 'shortPassSuccess']].sum().reset_index()
shortPass_tmp_df = shortPass_df.groupby('matchResult')[['shortPassTry','shortPassSuccess']].mean().reset_index()
shortPass_2_df = encoded_2_df.groupby(['matchId', 'ouid', 'matchResult'])[['shortPassTry', 'shortPassSuccess']].sum().reset_index()
shortPass_tmp_2_df = shortPass_2_df.groupby('matchResult')[['shortPassTry','shortPassSuccess']].mean().reset_index()

# --- 페이지 구성 설정 ---
st.set_page_config(page_title="FC온라인 분석 대시보드", layout="wide")

# --- 사이드바 ---
# st.sidebar.title("📁 페이지 이동")
# page = st.sidebar.radio("이동할 페이지를 선택하세요:", ["🏠 프로젝트 소개", "📊 모델 실험", "⚽ 주요 변수 비교 분석"])
# 페이지 목록
pages = {
    "🏠 프로젝트 소개": "intro",
    "📊 모델 실험": "model",
    "⚽ 주요 변수 비교 분석": "analysis"
}

# 선택된 페이지 상태 유지
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "🏠 프로젝트 소개"

# 사이드바 타이틀
st.sidebar.markdown("""
                    ## FC온라인 데이터 분석 프로젝트
                    """)

# 버튼 스타일 네비게이션
for page_name in pages:
    if st.sidebar.button(page_name):
        st.session_state.selected_page = page_name

# 선택된 페이지 확인
page = st.session_state.selected_page

# --- 사이드바 하단 정보 ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 문상혁")
st.sidebar.markdown("""
                    📨 E-mail: harrym8n@naver.com \
                    
                    🔗 GitHub: [방문하기](https://github.com/harrym8n)
                    """)

# --- 공통 스타일 ---
sns.set_theme(style='whitegrid', font=font_name)

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

    with st.expander("결과", expanded=True):
        st.markdown("""
        - 모델 성능(AUC 0.7148, F1-score 0.673) / 주요 변수(스루패스, 짧은 패스, 유효 슈팅, 롱패스, 블록 수)
        """)

    with st.expander("성과", expanded=True):
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
    st.title("⚽ 주요 변수 비교 분석")

    selected_var = st.selectbox("변수를 선택하세요", ["스루패스", "짧은패스", "유효슛", "롱패스", "블록"])

    if selected_var == "스루패스":
        st.markdown("### 스루패스 관련 비교 분석")

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



    elif selected_var == "짧은패스":
        st.markdown("### 1. 경기 결과별 평균 짧은 패스 데이터 비교 분석")

        # 평균 데이터 계산
        # 1) 다른 유저
        other_success_rate = sum(shortPass_tmp_df['shortPassSuccess']) / sum(shortPass_tmp_df['shortPassTry'])
        other_avg_try = shortPass_tmp_df['shortPassTry'].mean()
        other_avg_success = shortPass_tmp_df['shortPassSuccess'].mean()
        other_summary_df = pd.DataFrame({
            '구분': ['다른 유저'],
            '평균 시도 수': [round(other_avg_try, 2)],
            '평균 성공 수': [round(other_avg_success, 2)],
            '성공률': [round(other_success_rate, 4)]
        })

        # 2) 나
        my_success_rate = sum(shortPass_tmp_2_df['shortPassSuccess']) / sum(shortPass_tmp_2_df['shortPassTry'])
        my_avg_try = shortPass_tmp_2_df['shortPassTry'].mean()
        my_avg_success = shortPass_tmp_2_df['shortPassSuccess'].mean()
        my_summary_df = pd.DataFrame({
            '구분': ['나'],
            '평균 시도 수': [round(my_avg_try, 2)],
            '평균 성공 수': [round(my_avg_success, 2)],
            '성공률': [round(my_success_rate, 4)]
        })

        # 스타일
        sns.set_theme(style='whitegrid', font=font_name)


        # 그래프 나란히 배치
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**📌 다른 유저의 경기 결과별 짧은패스 시도 및 성공**")
            fig1, axes1 = plt.subplots(1, 2, figsize=(6, 3.5), sharex=True, sharey=True)
            sns.barplot(data=shortPass_tmp_df, x='matchResult', y='shortPassTry', hue='matchResult', ax=axes1[0])
            axes1[0].set_title('시도')
            sns.barplot(data=shortPass_tmp_df, x='matchResult', y='shortPassSuccess', hue='matchResult', ax=axes1[1])
            axes1[1].set_title('성공')
            for ax in axes1: ax.set_ylabel('')
            plt.ylim(60)
            plt.tight_layout()
            st.pyplot(fig1)
            st.markdown("**📊 다른 유저 평균 짧은패스 데이터**")
            st.dataframe(other_summary_df, use_container_width=True)

        with col2:
            st.markdown("**📌 나의 경기 결과별 짧은패스 시도 및 성공**")
            fig2, axes2 = plt.subplots(1, 2, figsize=(6, 3.5), sharex=True, sharey=True)
            sns.barplot(data=shortPass_tmp_2_df, x='matchResult', y='shortPassTry', hue='matchResult', ax=axes2[0])
            axes2[0].set_title('시도')
            sns.barplot(data=shortPass_tmp_2_df, x='matchResult', y='shortPassSuccess', hue='matchResult', ax=axes2[1])
            axes2[1].set_title('성공')
            for ax in axes2: ax.set_ylabel('')
            plt.ylim(10)
            plt.tight_layout()
            st.pyplot(fig2)
            st.markdown("**📊 나의 평균 짧은패스 데이터**")
            st.dataframe(my_summary_df, use_container_width=True)
        
        # 결론 정리
        with st.expander("🔍 결론 보기", expanded=True):
            st.markdown("""
            1. 나는 **짧은 패스를 잘하지만, 이것만으로 승리는 어렵다.**  
                - 패한 경기에서도 짧은 패스 성공률은 매우 높았음  
                - 승리한 유저들의 짧은 패스 수치는 크게 높지 않음 → **“짧은 패스 수”가 승리의 직접 요인은 아닐 수 있음**
            2. **패스 이후의 연결(슈팅, 공간 창출 등)이 중요할 수 있다.**  
                - 짧은 패스는 빌드업의 한 수단일 뿐이고, 그 이후 단계가 부족했을 가능성  
                - 유효 슛, 스루패스 성공 등과의 연계를 함께 분석해볼 필요 존재
            3. **과도한 짧은 패스는 오히려 템포를 느리게 할 수 있다.**  
                - 패한 경기에서도 패스 수치가 더 높다.  
                - 승리자보다 지나치게 많은 짧은 패스를 시도 → 공격 전개 속도가 느려지거나 턴오버, 태클 등으로 공 소유권이 넘어 갈 수 있음  
                - 공격 전개 속도 저하 → 수비에게 정비 시간 제공 → 슛 찬스 질 저하
            """)

        # 구분선
        st.markdown("---")
        st.markdown("### 2. 짧은 패스 수와 유효슛 수의 관계 분석")

        # col3 좌우 배치
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**다른 유저**")
            fig3 = sns.lmplot(
                data=encoded_df,
                x='shortPassTry', y='effectiveShootTotal', hue='matchResult',
                aspect=1.2, height=4, scatter_kws={'s': 15}
            )
            plt.title('짧은 패스 시도 수와 유효슛 수의 관계')
            plt.xlabel('짧은 패스 시도 수')
            plt.ylabel('유효슛 수')
            sns.despine()
            plt.grid(axis='x')
            st.pyplot(fig3)

            # 상관계수 계산 및 표시
            correlation_others = encoded_df[['shortPassTry', 'effectiveShootTotal']].corr().iloc[0, 1]
            st.markdown(f"📌 **상관계수**: {correlation_others:.4f}")

            # 결론
            with st.expander("🔍 결론 보기", expanded=True):
                st.markdown("""
                - 다른 유저들은 짧은 패스와 유효슛이 미세하지만 **양의 상관관계**를 보이며,  
                **승리한 유저는 음의 상관관계**, **패배한 유저는 양의 상관관계**가 나타남.
                """)

        with col_right:
            st.markdown("**나**")
            fig4 = sns.lmplot(
                data=encoded_2_df,
                x='shortPassTry', y='effectiveShootTotal', hue='matchResult',
                aspect=1.2, height=4, scatter_kws={'s': 15}
            )
            plt.title('짧은 패스 시도 수와 유효슛 수의 관계')
            plt.xlabel('짧은 패스 시도 수')
            plt.ylabel('유효슛 수')
            sns.despine()
            plt.grid(axis='x')
            st.pyplot(fig4)

            # 상관계수 계산 및 표시
            correlation_mine = encoded_2_df[['shortPassTry', 'effectiveShootTotal']].corr().iloc[0, 1]
            st.markdown(f"📌 **상관계수**: {correlation_mine:.4f}")

            # 결론
            with st.expander("🔍 결론 보기", expanded=True):
                st.markdown("""
                - 나는 **승리를 제외한 모든 결과에서 음의 상관관계**가 나타남.  
                즉, **짧은 패스를 선호하고 많이 하지만**, **유효슛으로 잘 이어지지 않음**.  
                빌드업 이후 마무리 단계의 개선 필요.
                """)

        # 구분선
        st.markdown("---")
        st.markdown("### 3. 짧은 패스 후 전진 패스 비율 계산")
        st.markdown("""
                    - 공격 템포 지표 정의 : 짧은패스 당 전진패스(스루패스, 롱패스)
                    - 공격 템포 = (스루패스 성공 + 롱패스 성공) / 짧은패스 성공
                    - 짧은 패스를 한 후 전진하는 스루패스/롱패스가 얼마나 이어지는지 비율로 측정")                
                    """)
    
        # col3 좌우 배치
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**다른 유저**")
            tmp_df = encoded_df.copy()
            tmp_df['attackTempo'] = (tmp_df['throughPassSuccess'] + tmp_df['longPassSuccess']) / tmp_df['shortPassSuccess']
            tempo_by_result = tmp_df.groupby('matchResult')['attackTempo'].mean().reset_index()

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(data=tempo_by_result, x='matchResult', y='attackTempo', hue='matchResult', ax=ax)
            ax.set_title('경기 결과별 공격 템포 (전진 패스 비율)')
            ax.set_xlabel('경기 결과')
            ax.set_ylabel('공격 템포 지표')
            sns.despine()
            st.pyplot(fig)

            # 전체 공격 템포
            tmp_df = tmp_df[tmp_df['matchResult']=='승']
            my_attack_tempo = round(tmp_df['attackTempo'].mean(),4)
            st.markdown(f"승리한 유저들의 평균 공격템포: {my_attack_tempo}")

            # 결론
            with st.expander("🔍 결론 보기", expanded=True):
                st.markdown("""
                - 다른 유저들은 짧은 패스와 유효슛이 미세하지만 **양의 상관관계**를 보이며,  
                **승리한 유저는 음의 상관관계**, **패배한 유저는 양의 상관관계**가 나타남.
                """)

        with col_right:
            st.markdown("**나**")
            tmp_df = encoded_2_df.copy()
            tmp_df['attackTempo'] = (tmp_df['throughPassSuccess'] + tmp_df['longPassSuccess']) / tmp_df['shortPassSuccess']
            tempo_by_result = tmp_df.groupby('matchResult')['attackTempo'].mean().reset_index()

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(data=tempo_by_result, x='matchResult', y='attackTempo', hue='matchResult', ax=ax)
            ax.set_title('경기 결과별 공격 템포 (전진 패스 비율)')
            ax.set_xlabel('경기 결과')
            ax.set_ylabel('공격 템포 지표')
            sns.despine()
            st.pyplot(fig)

            # 전체 공격 템포
            my_attack_tempo = round(tmp_df['attackTempo'].mean(),4)
            st.markdown(f"전체 평균 공격템포: {my_attack_tempo}")
            
            # 결론
            with st.expander("🔍 결론 보기", expanded=True):
                st.markdown("""
                1. 승리하는 경기에서 공격 템포(짧은 패스 당 전진패스)가 높다. - 공격 템포가 승리에 유의미히디.
                2. 다른 유저들은 짧은 패스 후 전진 패스가 비교적 활발하다. (공격 템포가 좋음)
                3. 나는 짧은 패스 후 전진 패스가 비교적 활발하지 않다. (공격 템포가 나쁨) 다른 유저 대비 약 0.5배
                """)


        # 인사이트
        st.markdown("---")
        st.markdown("### 🧠 인사이트")
        st.markdown("""
            - 승리한 유저의 짧은 패스 시도 및 성공 횟수가 패배/무승부보다 더 많음  
            - 짧은 패스 성공률은 모든 결과 그룹에서 큰 차이가 없으나, 시도 수 자체의 차이가 존재  
            - 내 데이터에서도 승리한 경기에서 짧은 패스 횟수가 유의하게 많았으나 공격 템포(전진패스 비율)가 승리 유저보다 낮았다.
            - **따라서, 짧은 패스 후 전진패스로 연결하는 공격 전개가 필요하다.**
            """)
        # ⚽ 유효슛 분석 결과
        ## ✅ 1. 경기 결과별 유효슛 수 평균 비교

    if selected_var == "유효슛":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 1. 다른 유저의 경기결과별 유효슛 분포 시각화")
            shoot_by_result = encoded_df.groupby('matchResult')['effectiveShootTotal'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(5, 5.27))
            sns.violinplot(data=encoded_df, x='matchResult', y='effectiveShootTotal', hue='matchResult', ax=ax)
            ax.set_title('경기결과별 유효슛 수 분포')
            ax.set_xlabel('경기 결과')
            ax.set_ylabel('유효슛 수')
            sns.despine()
            st.pyplot(fig)

            with st.expander("🔍 결론 보기", expanded=True):
                st.markdown("""
                - 유효슛은 경기결과와 상관관계가 높다.
                """)

        with col2:

            ## ✅ 2. 유효슛과 다른 변수와의 상관관계 분석

            st.markdown("### 2. 유효슛과 다른 변수와의 상관관계 분석")

            corr_df = encoded_df.drop(columns=['ouid', 'matchId', 'weekend', 'weekday', 'matchResult'])
            corr_matrix = corr_df.corr()[['effectiveShootTotal']]

            fig, ax = plt.subplots(figsize=(6, 8))
            sns.heatmap(corr_matrix.sort_values(by='effectiveShootTotal', ascending=False), annot=True, cmap='coolwarm', center=0, linewidths=0.5, linecolor='gray', ax=ax)
            plt.title('상관관계 시각화')
            plt.grid(False)
            st.pyplot(fig)

            with st.expander("🔍 결론 보기", expanded=True):
                st.markdown("""                        
                - 유효슛과 상관관계가 높은 변수 선별(corr > 0.3)
                    - 총 슛 횟수는 너무 당연한 결과로 분석가치가 높지 않아 제외
                    - 평균 평점, 골은 경기 결과 지표이기 때문에 경기 결과에 영향을 미칠 수 없어 제외
                    - 코너킥 횟수 관련 분석 진행 👇
                """)

        
        st.markdown("---")
        st.markdown("### 3. 경기 결과별 코너킥 횟수 분석")
        ## ✅ 3. 경기 결과별 코너킥 횟수 분석

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**다른 유저**")
            cornerKick_df = encoded_df.groupby(['matchId', 'ouid', 'matchResult'])[['cornerKick']].sum().reset_index()
            cornerKick_tmp_df = cornerKick_df.groupby('matchResult')[['cornerKick']].mean().reset_index()
            fig, ax = plt.subplots(figsize=(5.5, 4))
            sns.barplot(data=cornerKick_tmp_df, x='matchResult', y='cornerKick', hue='matchResult', ax=ax)
            ax.set_title("경기 결과별 평균 코너킥 횟수")
            ax.set_xlabel("경기 결과")
            ax.set_ylabel("코너킥")
            ax.set_ylim(1.00)
            sns.despine()
            st.pyplot(fig)

        with col2:
            st.markdown("**나**")
            cornerKick_df = encoded_2_df.groupby(['matchId', 'ouid', 'matchResult'])[['cornerKick']].sum().reset_index()
            cornerKick_tmp_2_df = cornerKick_df.groupby('matchResult')[['cornerKick']].mean().reset_index()
            fig, ax = plt.subplots(figsize=(5.5, 4))
            sns.barplot(data=cornerKick_tmp_2_df, x='matchResult', y='cornerKick', hue='matchResult', ax=ax)
            ax.set_title("경기 결과별 평균 코너킥 횟수")
            ax.set_xlabel("경기 결과")
            ax.set_ylabel("코너킥")
            ax.set_ylim(1.00)
            sns.despine()
            st.pyplot(fig)

        # 전체 평균
        avg_ck = round(encoded_2_df['cornerKick'].mean(), 4)
        st.markdown(f"내 전체 평균 코너킥 횟수: {avg_ck}")

        with st.expander("🔍 결론 보기", expanded=True):
            st.markdown("""            
            - 승리 유저의 평균 코너킥 횟수는 1.91, 나는 1.48로 승리 유저보다 코너킥 횟수가 적다.
            - 코너킥을 많이 얻을수록 유효슛 기회가 많아지고, 유효슛은 승리와 높은 상관관계를 보이므로 공격 진영 사이드로의 롱패스를 활용하여 코너킥 기회를 많이 창출해야 한다.
                - (롱패스와 코너킥 상관관계 높음)
            """)

        # 인사이트
        st.markdown("---")
        st.markdown("""
        ### 🧠 인사이트
        - 유효 슛 수는 승리한 유저가 압도적으로 많음  
        - 내 경기 데이터 분석에서도 유효 슛 수가 많을수록 승률이 높았음  
        - 머신러닝 모델에서 유효슛은 예측 성능에 중요한 변수 중 하나로 작용함  
        - 측면으로의 전진패스를 활용하여 코너킥 기회를 많이 창출해야 한다.
        """)

    if selected_var == "롱패스":
        # 1. 경기 결과별 평균 롱패스 시도 및 성공 수 시각화
        st.markdown("### 1. 경기 결과별 평균 롱패스 데이터")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 다른 유저 데이터")
            longPass_df = encoded_df.groupby(['matchId', 'ouid', 'matchResult'])[['longPassTry', 'longPassSuccess']].sum().reset_index()
            longPass_tmp_df = longPass_df.groupby('matchResult')[['longPassTry','longPassSuccess']].mean().reset_index()

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.barplot(data=longPass_tmp_df, x='matchResult', y='longPassTry', hue='matchResult', ax=axes[0])
            axes[0].set_title('롱패스 시도')

            sns.barplot(data=longPass_tmp_df, x='matchResult', y='longPassSuccess', hue='matchResult', ax=axes[1])
            axes[1].set_title('롱패스 성공')
            st.pyplot(fig)

        with col2:
            st.subheader("📊 내 데이터")
            longPass_2_df = encoded_2_df.groupby(['matchId', 'ouid', 'matchResult'])[['longPassTry', 'longPassSuccess']].sum().reset_index()
            longPass_tmp_2_df = longPass_2_df.groupby('matchResult')[['longPassTry','longPassSuccess']].mean().reset_index()

            fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
            sns.barplot(data=longPass_tmp_2_df, x='matchResult', y='longPassTry', hue='matchResult', ax=axes2[0])
            axes2[0].set_title('롱패스 시도')

            sns.barplot(data=longPass_tmp_2_df, x='matchResult', y='longPassSuccess', hue='matchResult', ax=axes2[1])
            axes2[1].set_title('롱패스 성공')
            st.pyplot(fig2)

        with st.expander("### 🔍 결론 보기", expanded=True):
            st.markdown("""
            - **승리한 유저**의 평균 롱패스 시도, 성공, 성공률이 **비기고 패한 유저보다 높다**.
            - 롱패스는 실제로 유효슛으로 이어지며, 유효슛은 승리와 높은 상관관계를 가진다.
            - 따라서 **롱패스를 통해 공격 기회를 적극적으로 창출**하는 것이 중요하다.
            """)

        # 2. 롱패스 성공률, 평균 비교
        st.markdown("### 2. 승리 유저와 내 롱패스 성공률 비교")

        # 다른 유저 성공률 계산
        tmp = longPass_tmp_df.copy()
        tmp = tmp[tmp['matchResult'] == '승']
        other_success_rate = round(sum(longPass_df['longPassSuccess']) / sum(longPass_df['longPassTry']), 4)
        other_avg_try = round(longPass_df['longPassTry'].mean(), 2)
        other_avg_success = round(longPass_df['longPassSuccess'].mean(), 2)

        # 내 성공률 계산
        my_success_rate = round(sum(longPass_2_df['longPassSuccess']) / sum(longPass_2_df['longPassTry']), 4)
        my_avg_try = round(longPass_2_df['longPassTry'].mean(), 2)
        my_avg_success = round(longPass_2_df['longPassSuccess'].mean(), 2)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### 📈 다른 유저")
            st.metric("롱패스 성공률", f"{other_success_rate*100:.2f}%")
            st.metric("평균 시도 수", f"{other_avg_try}")
            st.metric("평균 성공 수", f"{other_avg_success}")

        with col4:
            st.markdown("#### 📈 내 데이터")
            st.metric("롱패스 성공률", f"{my_success_rate*100:.2f}%")
            st.metric("평균 시도 수", f"{my_avg_try}")
            st.metric("평균 성공 수", f"{my_avg_success}")

        with st.expander("🔍 결론 보기", expanded=True):
            st.markdown(f"""
            - **나는 롱패스 성공률은 높지만**, 평균 시도 횟수는 승리한 유저보다 약 **20% 적음**.
            - 나는 **롱패스를 잘하는 유저**로 판단되며, 유효슛과 승리에 긍정적인 영향을 주는 롱패스를 **전략적으로 더 많이 시도**해야 한다.
            """)

    if selected_var == "블록":
        ### 1. 블록 성공 수 / 블록 성공률과 상대 유효슛 수의 상관관계
        st.markdown("### 1. 블록 성공 수 / 블록 성공률과 상대 유효슛 수의 상관관계")

        # 데이터 준비
        lose_df = encoded_df[encoded_df['matchResult'] == '패'][['matchId','effectiveShootTotal', 'possession']]
        lose_df = lose_df.rename(columns={'effectiveShootTotal':'lose_effectiveShootTotal', 'possession':'lose_possession'})

        win_df = encoded_df[encoded_df['matchResult'] == '승'][['matchId','blockTry','blockSuccess','possession']]
        win_df = win_df.rename(columns={'blockTry':'win_blockTry', 'blockSuccess':'win_blockSuccess', 'possession':'win_possession'})

        merge_df = pd.merge(lose_df, win_df, how='inner', on='matchId')
        merge_df['block_success_rate'] = merge_df['win_blockSuccess'] / merge_df['win_blockTry'].replace(0, 1)

        # 상관계수 계산
        corr1, p1 = pearsonr(merge_df['win_blockSuccess'], merge_df['lose_effectiveShootTotal'])
        corr2, p2 = pearsonr(merge_df['block_success_rate'], merge_df['lose_effectiveShootTotal'])

        # 상관계수 요약표
        corr_df = pd.DataFrame({
            '변수': ['블록 성공 수', '블록 성공률'],
            '상관계수': [round(corr1, 4), round(corr2, 4)],
            'p-value': [round(p1, 4), round(p2, 4)]
        })

        # 시각화 + 요약
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.set_style("whitegrid")

        # 그래프 꾸미기
        sns.regplot(
            data=merge_df,
            x='win_blockSuccess',
            y='lose_effectiveShootTotal',
            ax=ax1,
            scatter_kws={'color': '#1f77b4', 's': 30, 'alpha': 0.6},  
            line_kws={'color': '#ff7f0e', 'linewidth': 2}             
        )

        ax1.set_title('블록 성공 수 vs 상대 유효슛 수', fontsize=10, fontweight='bold')
        ax1.set_xlabel('블록 성공 수', fontsize=8)
        ax1.set_ylabel('상대 유효슛 수', fontsize=8)
        ax1.grid(axis='x')
        ax1.tick_params(labelsize=8)
        ax1.set_facecolor("#f9f9f9")  

        # 시각화 출력
        st.pyplot(fig1, use_container_width=False)

        # 상관계수 요약 표
        st.markdown("#### 상관계수 요약")
        st.dataframe(corr_df, use_container_width=True)

        ### 2. 경기 결과별 평균 블록 비교 분석
        st.markdown("### 2. 경기 결과별 평균 블록 비교 분석")

        block_df = encoded_df.groupby(['matchId', 'ouid', 'matchResult'])[['blockTry', 'blockSuccess']].sum().reset_index()
        block_tmp_df = block_df.groupby('matchResult')[['blockTry','blockSuccess']].mean().reset_index()

        block_2_df = encoded_2_df.groupby(['matchId', 'ouid', 'matchResult'])[['blockTry', 'blockSuccess']].sum().reset_index()
        block_tmp_2_df = block_2_df.groupby('matchResult')[['blockTry','blockSuccess']].mean().reset_index()

        # 시각화
        col3, col4 = st.columns(2)
        plt.rcParams['axes.unicode_minus'] = False

        with col3:
            st.markdown("**📌다른 유저**")
            fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3))
            sns.barplot(data=block_tmp_df, x='matchResult', y='blockTry', ax=axes2[0], hue='matchResult')
            axes2[0].set_title('블록 시도')
            sns.barplot(data=block_tmp_df, x='matchResult', y='blockSuccess', ax=axes2[1], hue='matchResult')
            axes2[1].set_title('블록 성공')
            st.pyplot(fig2)

        with col4:
            st.markdown("**📌내 데이터**")
            fig3, axes3 = plt.subplots(1, 2, figsize=(8, 3))
            sns.barplot(data=block_tmp_2_df, x='matchResult', y='blockTry', ax=axes3[0], hue='matchResult')
            axes3[0].set_title('블록 시도')
            sns.barplot(data=block_tmp_2_df, x='matchResult', y='blockSuccess', ax=axes3[1], hue='matchResult')
            axes3[1].set_title('블록 성공')
            st.pyplot(fig3)

        ### 3. 승리 경기 평균 블록 성공률
        tmp1 = block_tmp_df[block_tmp_df['matchResult'] == '승']
        success_rate_1 = sum(tmp1['blockSuccess']) / sum(tmp1['blockTry'])
        avg_try_1 = tmp1['blockTry'].mean()
        avg_success_1 = tmp1['blockSuccess'].mean()

        success_rate_2 = sum(block_tmp_2_df['blockSuccess']) / sum(block_tmp_2_df['blockTry'])
        avg_try_2 = block_tmp_2_df['blockTry'].mean()
        avg_success_2 = block_tmp_2_df['blockSuccess'].mean()

        summary_df = pd.DataFrame({
            '구분': ['승리 유저', '내 데이터'],
            '평균 블록 시도 수': [round(avg_try_1, 2), round(avg_try_2, 2)],
            '평균 블록 성공 수': [round(avg_success_1, 2), round(avg_success_2, 2)],
            '블록 성공률': [round(success_rate_1, 4), round(success_rate_2, 4)]
        })

        st.dataframe(summary_df, use_container_width=True)

        ### 4. 블록과 승패와의 회귀분석
        st.markdown("### 3. 블록과 승패와의 회귀분석")

        reg_df = encoded_df.copy()
        reg_df['win'] = reg_df['matchResult'].apply(lambda x: 1 if x == '승' else 0)
        reg_df['blockSuccessRate'] = reg_df['blockSuccess'] / reg_df['blockTry'].replace(0, 1)
        reg_df['blockSuccess_possession'] = reg_df['blockSuccess'] * reg_df['possession']

        X = reg_df[['blockTry', 'blockSuccess', 'blockSuccessRate', 'blockSuccess_possession']]
        y = reg_df['win']
        X_const = sm.add_constant(X)
        logit_model = sm.Logit(y, X_const)
        result = logit_model.fit(disp=0)

        # 회귀 요약표
        summary_table = pd.DataFrame({
            '변수명': result.params.index,
            '회귀계수': result.params.values,
            'p-value': result.pvalues
        }).round(4)

        # 평균 Marginal Effect
        marginal_effect = result.get_margeff(at='overall').summary_frame().round(4).reset_index()
        marginal_effect.columns = ['변수명', 'Marginal Effect', 'Std Err', 'z', 'P>|z|', '[0.025', '0.975]']

        # 병합표
        final_df = pd.merge(summary_table, marginal_effect[['변수명', 'P>|z|', 'Marginal Effect']], on='변수명', how='left')

        # 해석 열 추가
        final_df['해석'] = [
            '상수항',
            '블록 시도가 1회 증가할 때, 승리 확률은 약 1.63% 감소',
            '블록 성공이 1회 증가할 때, 승리 확률은 약 81.86% 감소',
            '블록 성공률과 승리 확률 사이의 유의미한 관계 없음',
            '점유율을 동반한 블록 성공이 1회 증가할 때, 승리 확률은 약 1.74% 증가'
        ]

        # 열 순서 재정렬
        final_df = final_df[['변수명', 'P>|z|', 'Marginal Effect', '해석']]
        final_df.columns = ['변수명', 'P-value', 'Marginal Effect', '해석']
        st.dataframe(final_df, use_container_width=True)

        st.markdown("---")

       # 해석 및 결론
        st.markdown("### 🔍 결론 보기")
        with st.expander("🔽 펼치기", expanded=True):
            st.markdown("""
            1. **무조건적인 블록 시도/성공은 승리에 도움이 되지 않음**
                - 블록 횟수가 많을수록 패배 가능성이 높음 → 수세적 운영을 시사  
            2. **효율보다 '상황'이 중요함**
                - `블록 성공 * 점유율`이 승리와 유의미한 양의 관계
                - 공을 소유하며 블록에 성공해야 경기 흐름을 통제 가능
                        """)
        st.markdown("""
            ### 🧠 인사이트
            1. 수세적인 플레이 줄이기 → 블록만 많은 경기는 오히려 패배로 이어질 수 있음
            2. 전방 압박 및 점유율 회복 전술 고려 → 공수 전환이 빠른 적극적 수비가 승리에 긍정적
            """)