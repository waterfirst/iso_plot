import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata
import io

# 페이지 설정
st.set_page_config(
    page_title="ISO Polar Plot Visualization", 
    page_icon="📊",
    layout="wide"
)

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = ['DejaVu Sans',  'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_and_process_data(uploaded_file):
    """CSV 파일을 로드하고 처리하는 함수"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # 데이터 구조 확인
        if 'Theta' not in df.columns:
            st.error("CSV 파일에 'Theta' 컬럼이 없습니다.")
            return None
        
        # Phi 컬럼들 (숫자로 된 컬럼명) 찾기
        phi_columns = [col for col in df.columns if col != 'Theta' and col.replace('.', '').isdigit()]
        phi_values = [float(col) for col in phi_columns]
        
        # 데이터를 long format으로 변환
        df_long = df.melt(id_vars=['Theta'], 
                         value_vars=phi_columns, 
                         var_name='Phi', 
                         value_name='Luminance')
        
        df_long['Phi'] = df_long['Phi'].astype(float)
        df_long['Theta'] = df_long['Theta'].astype(float)
        df_long['Luminance'] = df_long['Luminance'].astype(float)
        
        return df_long, phi_values
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

def polar_to_cartesian(theta, phi, r_max=1):
    """극좌표를 직교좌표로 변환"""
    # Theta는 반지름 (정규화), Phi는 각도
    r = theta / theta.max() * r_max if theta.max() > 0 else 0
    phi_rad = np.radians(phi)
    
    x = r * np.cos(phi_rad)
    y = r * np.sin(phi_rad)
    
    return x, y


def create_matplotlib_polar_plot(df_long):
    """Matplotlib으로 ISO polar plot 생성"""
    
    # 데이터 준비
    theta_vals = df_long['Theta'].unique()
    phi_vals = df_long['Phi'].unique()
    
    # 격자 생성
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    
    # Luminance 값을 2D 배열로 재구성
    luminance_grid = np.zeros_like(theta_grid)
    
    for i, phi in enumerate(phi_vals):
        for j, theta in enumerate(theta_vals):
            mask = (df_long['Phi'] == phi) & (df_long['Theta'] == theta)
            if mask.any():
                luminance_grid[i, j] = df_long.loc[mask, 'Luminance'].iloc[0]
    
    # 극좌표계 설정
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    # 각도를 라디안으로 변환
    phi_rad = np.radians(phi_grid)
    
    # 컨투어 플롯 생성
    contour = ax.contourf(phi_rad, theta_grid, luminance_grid, 
                         levels=20, cmap='jet', alpha=0.8)
    
    # 컨투어 라인 추가
    contour_lines = ax.contour(phi_rad, theta_grid, luminance_grid, 
                              levels=10, colors='white', alpha=0.5, linewidths=0.5)
    
    # 축 설정
    ax.set_ylim(0, theta_vals.max())
    ax.set_theta_zero_location('E')  # 0도를 오른쪽으로
    ax.set_theta_direction(1)  # 시계 반대 방향
    
    # 각도 눈금 설정
    ax.set_thetagrids(range(0, 360, 30))
    
    # 반지름 눈금 설정
    ax.set_ylim(0, theta_vals.max())
    ax.set_yticks(theta_vals)
    ax.set_yticklabels([f'{int(t)}°' for t in theta_vals])
    
    # 컬러바 추가
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Luminance', fontsize=12)
    
    # 제목
    ax.set_title('ISO Luminance Distribution', fontsize=14, pad=20)
    
    return fig

def create_matplotlib_cartesian_plot(df_long):
    """Matplotlib으로 직교좌표계 ISO plot 생성"""
    
    # 극좌표를 직교좌표로 변환
    x, y = polar_to_cartesian(df_long['Theta'], df_long['Phi'])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 삼각 보간을 위한 격자 생성
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # 보간
    zi = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), method='cubic')
    
    # 원형 마스크 생성
    center = (0, 0)
    radius = 1
    mask = (xi_grid - center[0])**2 + (yi_grid - center[1])**2 <= radius**2
    zi_masked = np.where(mask, zi, np.nan)
    
    # 컨투어 플롯
    contour = ax.contourf(xi_grid, yi_grid, zi_masked, levels=20, cmap='jet')
    contour_lines = ax.contour(xi_grid, yi_grid, zi_masked, levels=10, 
                              colors='white', alpha=0.7, linewidths=0.5)
    
    # 원형 경계 그리기
    circle = patches.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    # 각도 라벨 추가
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in angles:
        x_pos = 1.1 * np.cos(np.radians(angle))
        y_pos = 1.1 * np.sin(np.radians(angle))
        ax.text(x_pos, y_pos, f'{angle}°', ha='center', va='center', fontsize=10)
    
    # 반지름 원들 그리기
    theta_vals = df_long['Theta'].unique()
    for theta in theta_vals[1:]:  # 0은 제외
        r_norm = theta / theta_vals.max()
        circle_r = patches.Circle((0, 0), r_norm, fill=False, color='white', 
                                 alpha=0.5, linewidth=0.5)
        ax.add_patch(circle_r)
    
    # 축 설정
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 컬러바
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Luminance', fontsize=12)
    
    # 제목
    ax.set_title('ISO Luminance Distribution', fontsize=14, pad=20)
    
    return fig

def create_matplotlib_polar_plot(df_long):
    """개선된 Matplotlib ISO polar plot (자연스러운 gradient, 등고선 제거)"""
    
    # 데이터 준비
    theta_vals = df_long['Theta'].unique()
    phi_vals = df_long['Phi'].unique()
    
    # 더 조밀한 격자 생성 (보간을 위해)
    theta_interp = np.linspace(theta_vals.min(), theta_vals.max(), 100)
    phi_interp = np.linspace(phi_vals.min(), phi_vals.max(), 360)
    
    # 원본 데이터를 2D 배열로 재구성
    df_pivot = df_long.pivot(index='Theta', columns='Phi', values='Luminance')
    
    # 보간을 위한 함수 생성
    from scipy.interpolate import interp2d
    
    # 2D 보간 함수 생성
    f_interp = interp2d(df_pivot.columns, df_pivot.index, df_pivot.values, kind='cubic')
    
    # 보간된 격자 생성
    theta_grid, phi_grid = np.meshgrid(theta_interp, phi_interp)
    luminance_interp = f_interp(phi_interp, theta_interp)
    
    # 극좌표계 설정
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    # 각도를 라디안으로 변환
    phi_rad = np.radians(phi_grid)
    
    # 매우 세밀한 컨투어 플롯 생성 (더 많은 레벨로 부드러운 gradient)
    levels = np.linspace(df_long['Luminance'].min(), df_long['Luminance'].max(), 100)
    
    contour = ax.contourf(phi_rad, theta_grid, luminance_interp.T, 
                         levels=levels, 
                         cmap='jet', 
                         alpha=1.0,
                         antialiased=True)
    
    # 등고선 제거 (contour lines 없음)
    
    # 축 설정
    ax.set_ylim(0, theta_vals.max())
    ax.set_theta_zero_location('E')  # 0도를 오른쪽으로
    ax.set_theta_direction(1)  # 시계 반대 방향
    
    # 각도 눈금 설정 (30도 간격)
    theta_ticks = np.arange(0, 360, 30)
    ax.set_thetagrids(theta_ticks, [f'{int(t)}°' for t in theta_ticks])
    
    # 반지름 눈금 설정
    ax.set_rticks(theta_vals)
    ax.set_rlabel_position(45)  # 레이블 위치
    
    # 격자 스타일 개선
    ax.grid(True, alpha=0.3, color='white', linewidth=0.8)
    ax.set_facecolor('black')
    
    # 컬러바 추가 (위치와 크기 조정)
    cbar = plt.colorbar(contour, ax=ax, shrink=0.6, pad=0.08, aspect=30)
    cbar.set_label('Luminance', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    # 제목 스타일 개선
    ax.set_title('ISO Luminance Distribution', fontsize=16, pad=30, fontweight='bold')
    
    # 전체적인 스타일 개선
    plt.tight_layout()
    
    return fig


def create_matplotlib_cartesian_plot(df_long):
    """개선된 Matplotlib 직교좌표계 ISO plot (자연스러운 gradient)"""
    
    # 극좌표를 직교좌표로 변환
    x, y = polar_to_cartesian(df_long['Theta'], df_long['Phi'])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 더 조밀한 격자 생성 (해상도 증가)
    resolution = 200
    xi = np.linspace(x.min()-0.1, x.max()+0.1, resolution)
    yi = np.linspace(y.min()-0.1, y.max()+0.1, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # RBF (Radial Basis Function) 보간 사용 - 더 부드러운 결과
    from scipy.interpolate import Rbf
    
    # RBF 보간 함수 생성
    rbf = Rbf(x, y, df_long['Luminance'], function='multiquadric', smooth=0.1)
    zi = rbf(xi_grid, yi_grid)
    
    # 원형 마스크 생성
    center = (0, 0)
    radius = 1.05
    mask = (xi_grid - center[0])**2 + (yi_grid - center[1])**2 <= radius**2
    zi_masked = np.where(mask, zi, np.nan)
    
    # 매우 세밀한 컨투어 플롯 (등고선 없음)
    levels = np.linspace(np.nanmin(zi_masked), np.nanmax(zi_masked), 200)
    
    contour = ax.contourf(xi_grid, yi_grid, zi_masked, 
                         levels=levels, 
                         cmap='jet', 
                         alpha=1.0,
                         antialiased=True,
                         extend='both')
    
    # 원형 경계 그리기
    circle = patches.Circle((0, 0), 1, fill=False, color='white', linewidth=3)
    ax.add_patch(circle)
    
    # 각도 라벨 추가 (스타일 개선)
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in angles:
        x_pos = 1.15 * np.cos(np.radians(angle))
        y_pos = 1.15 * np.sin(np.radians(angle))
        ax.text(x_pos, y_pos, f'{angle}°', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 반지름 원들 그리기 (더 세밀하게)
    theta_vals = df_long['Theta'].unique()
    for theta in theta_vals[1:]:
        r_norm = theta / theta_vals.max()
        circle_r = patches.Circle((0, 0), r_norm, fill=False, color='white', 
                                 alpha=0.4, linewidth=1)
        ax.add_patch(circle_r)
        
        # 반지름 레이블 추가
        ax.text(r_norm, 0, f'{int(theta)}°', ha='center', va='bottom',
                fontsize=10, color='white', fontweight='bold')
    
    # 방향 선 그리기 (12방향)
    for angle in range(0, 360, 30):
        x_end = 1.05 * np.cos(np.radians(angle))
        y_end = 1.05 * np.sin(np.radians(angle))
        ax.plot([0, x_end], [0, y_end], 'w-', alpha=0.3, linewidth=1)
    
    # 축 설정
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('black')
    
    # 컬러바 (개선된 스타일)
    cbar = plt.colorbar(contour, ax=ax, shrink=0.6, aspect=30)
    cbar.set_label('Luminance', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    # 제목
    ax.set_title('ISO Luminance Distribution', fontsize=16, pad=30, 
                fontweight='bold', color='black')
    
    plt.tight_layout()
    
    return fig




def create_plotly_polar_plot(df_long, vmin, vmax, cmap='jet'):
    """Plotly polar plot with 사용자 정의 범위"""
    
    # Plotly 컬러맵 변환
    plotly_colorscales = {
        'jet': 'Jet',
        'viridis': 'Viridis',
        'plasma': 'Plasma',
        'inferno': 'Inferno',
        'hot': 'Hot',
        'cool': 'RdBu',
        'rainbow': 'Rainbow',
        'turbo': 'Turbo'
    }
    
    plotly_cmap = plotly_colorscales.get(cmap, 'Jet')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=df_long['Theta'],
        theta=df_long['Phi'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_long['Luminance'],
            colorscale=plotly_cmap,
            cmin=vmin,
            cmax=vmax,
            showscale=True,
            colorbar=dict(title="Luminance")
        ),
        text=df_long['Luminance'],
        hovertemplate='Theta: %{r}°<br>Phi: %{theta}°<br>Luminance: %{text:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Interactive ISO Luminance Distribution<br>(Range: {vmin:.1f} - {vmax:.1f})',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, df_long['Theta'].max()]
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 360, 30)),
                ticktext=[f'{i}°' for i in range(0, 360, 30)]
            )
        ),
        width=700,
        height=700
    )
    
    return fig

def create_plotly_heatmap(df_long):
    """Plotly로 히트맵 생성"""
    
    # 데이터 피벗
    df_pivot = df_long.pivot(index='Theta', columns='Phi', values='Luminance')
    
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns,
        y=df_pivot.index,
        colorscale='Jet',
        colorbar=dict(title="Luminance")
    ))
    
    fig.update_layout(
        title='Luminance Heatmap (Theta vs Phi)',
        xaxis_title='Phi (degrees)',
        yaxis_title='Theta (degrees)',
        width=800,
        height=600
    )
    
    return fig


def create_smooth_polar_plot(df_long, vmin, vmax, cmap='jet', resolution=300):
    """개선된 Smooth Polar Plot with 사용자 정의 컬러바 범위"""
    
    # 데이터 준비
    theta_vals = df_long['Theta'].unique()
    phi_vals = df_long['Phi'].unique()
    
    # 보간을 위한 격자 생성
    theta_interp = np.linspace(theta_vals.min(), theta_vals.max(), resolution//3)
    phi_interp = np.linspace(0, 360, resolution)
    
    # 원본 데이터를 2D 배열로 재구성
    df_pivot = df_long.pivot(index='Theta', columns='Phi', values='Luminance')
    
    # RegularGridInterpolator로 대체
    from scipy.interpolate import RegularGridInterpolator
    f_interp = RegularGridInterpolator(
        (df_pivot.index.values, df_pivot.columns.values),
        df_pivot.values,
        method='linear',  # 'cubic'도 가능하지만 grid가 충분히 조밀해야 함
        bounds_error=False,
        fill_value=np.nan
    )
    theta_grid, phi_grid = np.meshgrid(theta_interp, phi_interp, indexing='ij')
    points = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)
    luminance_interp = f_interp(points).reshape(theta_grid.shape)
    
    # 극좌표계 설정
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    # 각도를 라디안으로 변환
    phi_rad = np.radians(phi_grid)
    
    # 사용자 정의 범위로 레벨 생성
    levels = np.linspace(vmin, vmax, 200)
    
    # 컨투어 플롯 (사용자 정의 범위 적용)
    contour = ax.contourf(phi_rad, theta_grid, luminance_interp, 
                         levels=levels, 
                         cmap=cmap, 
                         alpha=1.0,
                         antialiased=True,
                         vmin=vmin,
                         vmax=vmax,
                         extend='both')  # 범위를 벗어나는 값들 처리
    
    # 축 설정
    ax.set_ylim(0, theta_vals.max())
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    
    # 각도 눈금 설정
    theta_ticks = np.arange(0, 360, 30)
    ax.set_thetagrids(theta_ticks, [f'{int(t)}°' for t in theta_ticks])
    
    # 반지름 눈금 설정
    #ax.set_rticks(theta_vals)
    #ax.set_rlabel_position(45)
    
    # 격자 스타일
    ax.grid(True, alpha=0.3, color='white', linewidth=0.8)
    ax.set_facecolor('black')
    
    # 컬러바 (사용자 정의 범위 표시)
    cbar = plt.colorbar(contour, ax=ax, shrink=0.6, pad=0.08, aspect=30)
    cbar.set_label('Luminance', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    # 컬러바 범위 설정
    cbar.set_ticks(np.linspace(vmin, vmax, 6))
    cbar.set_ticklabels([f'{val:.1f}' for val in np.linspace(vmin, vmax, 6)])
    
    # 제목
    ax.set_title(f'ISO Luminance Distribution\n(Range: {vmin:.1f} - {vmax:.1f})', 
                fontsize=16, pad=30, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_enhanced_cartesian_plot(df_long, vmin, vmax, cmap='jet', resolution=300):
    """개선된 Cartesian Plot with 사용자 정의 컬러바 범위"""
    
    # 극좌표를 직교좌표로 변환
    theta_rad = np.radians(df_long['Theta'])
    phi_rad = np.radians(df_long['Phi'])
    r_norm = df_long['Theta'] / df_long['Theta'].max()
    
    x = r_norm * np.cos(phi_rad)
    y = r_norm * np.sin(phi_rad)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 고해상도 격자
    xi = np.linspace(-1.1, 1.1, resolution)
    yi = np.linspace(-1.1, 1.1, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # RBF 보간
    from scipy.interpolate import Rbf
    rbf = Rbf(x, y, df_long['Luminance'], function='multiquadric', smooth=0.1)
    zi = rbf(xi_grid, yi_grid)
    
    # 원형 마스크
    mask = xi_grid**2 + yi_grid**2 <= 1.02**2
    zi_masked = np.where(mask, zi, np.nan)
    
    # 사용자 정의 범위로 레벨 생성
    levels = np.linspace(vmin, vmax, 200)
    
    # 컨투어 플롯
    contour = ax.contourf(xi_grid, yi_grid, zi_masked, 
                         levels=levels, 
                         cmap=cmap, 
                         alpha=1.0,
                         antialiased=True,
                         vmin=vmin,
                         vmax=vmax,
                         extend='both')
    
    # 원형 경계
    circle = patches.Circle((0, 0), 1, fill=False, color='white', linewidth=3)
    ax.add_patch(circle)
    
    # 각도 라벨
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in angles:
        x_pos = 1.15 * np.cos(np.radians(angle))
        y_pos = 1.15 * np.sin(np.radians(angle))
        ax.text(x_pos, y_pos, f'{angle}°', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 반지름 원들
    theta_vals = df_long['Theta'].unique()
    for theta in theta_vals[1:]:
        r_norm_circle = theta / theta_vals.max()
        circle_r = patches.Circle((0, 0), r_norm_circle, fill=False, 
                                 color='white', alpha=0.4, linewidth=1)
        ax.add_patch(circle_r)
        ax.text(r_norm_circle, 0, f'{int(theta)}°', ha='center', va='bottom',
                fontsize=10, color='white', fontweight='bold')
    
    # 축 설정
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('black')
    
    # 컬러바
    cbar = plt.colorbar(contour, ax=ax, shrink=0.6, aspect=30)
    cbar.set_label('Luminance', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(vmin, vmax, 6))
    cbar.set_ticklabels([f'{val:.1f}' for val in np.linspace(vmin, vmax, 6)])
    
    # 제목
    ax.set_title(f'ISO Luminance Distribution\n(Range: {vmin:.1f} - {vmax:.1f})', 
                fontsize=16, pad=30, fontweight='bold', color='black')
    
    plt.tight_layout()
    return fig

def create_colorbar_preview(vmin, vmax, cmap='jet'):
    """컬러바 미리보기 생성"""
    
    fig, ax = plt.subplots(figsize=(1, 4))
    
    # 컬러바만 표시
    gradient = np.linspace(vmin, vmax, 256).reshape(256, 1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # 축 설정
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 256)
    ax.set_xticks([])
    
    # Y축 눈금 (5개 정도)
    tick_positions = np.linspace(0, 256, 6)
    tick_labels = np.linspace(vmin, vmax, 6)
    #ax.set_yticks(tick_positions)
    #ax.set_yticklabels([f'{val:.1f}' for val in tick_labels])
    
    ax.set_ylabel('Luminance', fontsize=10)
    ax.set_title('Preview', fontsize=10)
    
    plt.tight_layout()
    return fig

def enhanced_polar_to_cartesian(df_long, resolution=200):
    """향상된 극좌표-직교좌표 변환 및 보간"""
    
    # 원본 극좌표를 직교좌표로 변환
    theta_rad = np.radians(df_long['Theta'])
    phi_rad = np.radians(df_long['Phi'])
    
    # 정규화된 반지름 (0-1)
    r_norm = df_long['Theta'] / df_long['Theta'].max()
    
    x = r_norm * np.cos(phi_rad)
    y = r_norm * np.sin(phi_rad)
    
    # 고해상도 격자 생성
    xi = np.linspace(-1.1, 1.1, resolution)
    yi = np.linspace(-1.1, 1.1, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # 여러 보간 방법 중 선택
    from scipy.interpolate import griddata
    
    # 'cubic' 보간으로 매우 부드러운 결과
    zi = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), 
                  method='cubic', fill_value=df_long['Luminance'].min())
    
    return xi_grid, yi_grid, zi

def main():
    st.title("📊 ISO Polar Plot Visualization")
    st.markdown("CSV 파일을 업로드하여 ISO(광학 강도 분포) polar plot을 생성합니다.")
    
    # 사이드바
    st.sidebar.header("설정")
    
    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader(
        "CSV 파일 선택", 
        type=['csv'],
        help="Theta 컬럼과 각도별 Phi 값들이 포함된 CSV 파일을 업로드하세요."
    )
    
    if uploaded_file is not None:
        # 데이터 로드
        result = load_and_process_data(uploaded_file)
        
        if result is not None:
            df_long, phi_values = result
            
            # 데이터 정보 표시
            st.sidebar.success("✅ 데이터 로드 성공!")
            st.sidebar.write(f"**데이터 포인트:** {len(df_long)}")
            st.sidebar.write(f"**Theta 범위:** {df_long['Theta'].min()}° - {df_long['Theta'].max()}°")
            st.sidebar.write(f"**Phi 범위:** {df_long['Phi'].min()}° - {df_long['Phi'].max()}°")
            
            # 데이터 범위 정보
            data_min = float(df_long['Luminance'].min())
            data_max = float(df_long['Luminance'].max())
            st.sidebar.write(f"**Luminance 범위:** {data_min:.2f} - {data_max:.2f}")
            
            st.sidebar.divider()
            
            # 컬러바 범위 설정
            st.sidebar.subheader("🎨 컬러바 설정")
            
            # 자동/수동 모드 선택
            colorbar_mode = st.sidebar.radio(
                "컬러바 범위 모드",
                ["자동 (데이터 범위)", "수동 설정"],
                help="자동: 데이터의 최소/최대값 사용, 수동: 직접 범위 설정"
            )
            
            if colorbar_mode == "자동 (데이터 범위)":
                vmin, vmax = data_min, data_max
                st.sidebar.info(f"자동 범위: {vmin:.2f} ~ {vmax:.2f}")
                
            else:  # 수동 설정
                col1, col2 = st.sidebar.columns(2)
                
                with col1:
                    vmin = st.number_input(
                        "최소값",
                        value=data_min,
                        step=0.1,
                        format="%.2f",
                        help="컬러바의 최소값을 설정합니다"
                    )
                
                with col2:
                    vmax = st.number_input(
                        "최대값",
                        value=data_max,
                        step=0.1,
                        format="%.2f",
                        help="컬러바의 최대값을 설정합니다"
                    )
                
                # 범위 검증
                if vmin >= vmax:
                    st.sidebar.error("최소값은 최대값보다 작아야 합니다!")
                    vmin, vmax = data_min, data_max
                
                # 리셋 버튼
                if st.sidebar.button("범위 초기화"):
                    st.rerun()
            
            # 컬러맵 선택
            colormap_options = ['jet', 'viridis', 'plasma', 'inferno', 'hot', 'cool', 'rainbow', 'turbo']
            selected_cmap = st.sidebar.selectbox(
                "컬러맵",
                colormap_options,
                index=0,
                help="컬러바의 색상 스타일을 선택합니다"
            )
            
            st.sidebar.divider()
            
            # 시각화 옵션
            plot_type = st.sidebar.selectbox(
                "시각화 타입",
                ["Smooth Polar Plot", "Enhanced Cartesian", "Plotly Interactive", "Plotly Heatmap"]
            )
            
            # 해상도 설정
            resolution = st.sidebar.slider(
                "해상도",
                min_value=100,
                max_value=500,
                value=300,
                step=50,
                help="높을수록 더 부드럽지만 느려집니다"
            )
            
            # 메인 컨텐츠
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if plot_type == "Smooth Polar Plot":
                    st.subheader("Smooth Polar Plot")
                    fig = create_smooth_polar_plot(df_long, vmin, vmax, selected_cmap, resolution)
                    st.pyplot(fig)
                    
                elif plot_type == "Enhanced Cartesian":
                    st.subheader("Enhanced Cartesian Plot")
                    fig = create_enhanced_cartesian_plot(df_long, vmin, vmax, selected_cmap, resolution)
                    st.pyplot(fig)
                    
                elif plot_type == "Plotly Interactive":
                    st.subheader("Plotly Interactive Plot")
                    fig = create_plotly_polar_plot(df_long, vmin, vmax, selected_cmap)
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type == "Plotly Heatmap":
                    st.subheader("Plotly Heatmap")
                    fig = create_plotly_heatmap(df_long, vmin, vmax, selected_cmap)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                #st.subheader("컬러바 미리보기")
                # 컬러바 미리보기 생성
                #preview_fig = create_colorbar_preview(vmin, vmax, selected_cmap)
                #st.pyplot(preview_fig)
                
                st.subheader("데이터 미리보기")
                st.dataframe(df_long.head(10))
                
                # 통계 정보
                st.subheader("통계 정보")
                st.write("**Luminance 통계:**")
                stats_df = pd.DataFrame({
                    '통계': ['최소값', '최대값', '평균', '표준편차', '중간값'],
                    '값': [
                        f"{df_long['Luminance'].min():.2f}",
                        f"{df_long['Luminance'].max():.2f}",
                        f"{df_long['Luminance'].mean():.2f}",
                        f"{df_long['Luminance'].std():.2f}",
                        f"{df_long['Luminance'].median():.2f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True)
            
 
            
            # ISO_PLOT 이미지 다운로드
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            st.download_button(
                label="ISO_PLOT 이미지 다운로드 (PNG)",
                data=img_buffer.getvalue(),
                file_name="iso_plot.png",
                mime="image/png"
            )
    
    else:
        # 샘플 데이터 및 사용법 안내
        st.info("👆 사이드바에서 CSV 파일을 업로드하세요.")
        
        st.subheader("📋 사용법")
        st.markdown("""
        1. **CSV 파일 형식**: 
           - 첫 번째 컬럼: `Theta` (각도 값)
           - 나머지 컬럼: 각 Phi 각도 (0, 10, 20, ..., 360)
           
        2. **지원하는 시각화**:
           - Matplotlib Polar: 전통적인 극좌표 플롯
           - Matplotlib Cartesian: 직교좌표계 ISO 플롯
           - Plotly Interactive: 인터랙티브 극좌표 플롯
           - Plotly Heatmap: 히트맵 형태의 데이터 시각화
        """)

if __name__ == "__main__":
    main()
