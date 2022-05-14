import pandas as pd
from patsy import dmatrices
from statsmodels.formula.api import ols # 회귀 분석 함수
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt # 그래프 
from matplotlib import font_manager, rc

# 그래프에서 한글 깨짐 문제를 해결하기 위한 코드
font_path = "C:/Windows/Fonts/KoPub Dotum Bold.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

area_list = ["종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구","강북구","도봉구","노원구","은평구","서대문구","마포구","양천구","강서구","구로구","금천구","영등포구","동작구","관악구","서초구","강남구","송파구","강동구"]

# 데이터 로드
# 종속변수: 사고 건수, 독립변수: crosswalk+accident_zone+protect_zone+population
df = pd.read_csv('dataset.csv', encoding="cp949")

# 그래프에 들어갈 값
plot_x = df["area"].tolist() # 이게 x축 기준 
plot_y = [] # y축에 그려질 값 - 각 동 별 (지표값m * 가중치m)의 합 을 리스트로

# 종속변수 ~ 독립변수의 형태로 모형식 작성
# 사고건수 ~ 보행등 + 사고 구역 + 노인인구수 + 노인보호구역 
formula = 'accident_cnt ~ crosswalk+accident_zone+protect_zone+population'

# VIF 계산
df_all_data = pd.DataFrame(df)
df_all_data.drop(columns=["area_big", "area", "signal"], inplace=True)

y, X = dmatrices(formula, data=df_all_data, return_type="dataframe")

vif = pd.DataFrame()
vif["vif_factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print("VIF calc result =======================================")
print(vif)

# 위에서 실시한 회귀분석에 대한 요약 결과 확인 가능
res = ols(formula, data=df_all_data).fit()
print("\nregression analysys result =======================================")
print(res.summary())

# 지표별 수치값 
crosswalk_estimate = res.params.crosswalk 
accident_zone_estimate = res.params.accident_zone 
protect_zone_estimate = res.params.protect_zone 
population_estimate = res.params.population 

for area in area_list:
    df_area = pd.DataFrame(df[df.area_big == area])
    df_area.drop(columns=["area_big", "area", "signal"], inplace=True)

    # 동 별 행정동 점수 계산
    for index, row in df_area.iterrows():
        plot_y.append(crosswalk_estimate*row.crosswalk + accident_zone_estimate*row.accident_zone + \
            protect_zone_estimate*row.protect_zone + population_estimate*row.population)


plot_data = [[plot_x[i], plot_y[i]] for i in range(len(plot_x))]
plot_data = sorted(plot_data, key=lambda x : x[1], reverse=True)

x_data = list(zip(*plot_data[:10]))[0]
y_data = list(zip(*plot_data[:10]))[1]

plt_bar = plt.bar(x_data, y_data, width=0.35, color="black", tick_label=x_data) 
plt.bar_label(plt_bar, label_type="edge")
plt.show()