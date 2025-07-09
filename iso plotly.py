import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import griddata, interp2d, RegularGridInterpolator, Rbf
import io
import base64

# 페이지 설정
st.set_page_config(
    page_title="ISO Polar Plot Visualization", 
    page_icon="📊",
    layout="wide"
)

# 한글 폰트 설정 (matplotlib - 크로스섹션용)
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

def create_plotly_smooth_polar_plot(df_long, vmin, vmax, cmap='Jet', resolution=300):
    """Plotly로 부드러운 극좌표 플롯 생성"""
    
    # 데이터 준비
    theta_vals = df_long['Theta'].unique()
    phi_vals = df_long['Phi'].unique()
    
    # 보간을 위한 격자 생성
    theta_interp = np.linspace(theta_vals.min(), theta_vals.max(), resolution//3)
    phi_interp = np.linspace(0, 360, resolution)
    
    # 원본 데이터를 2D 배열로 재구성
    df_pivot = df_long.pivot(index='Theta', columns='Phi', values='Luminance')
    
    # 보간
    try:
        f_interp = RegularGridInterpolator(
            (df_pivot.index.values, df_pivot.columns.values),
            df_pivot.values,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        theta_grid, phi_grid = np.meshgrid(theta_interp, phi_interp, indexing='ij')
        points = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)
        luminance_interp = f_interp(points).reshape(theta_grid.shape)
    except Exception as e:
        # 보간 실패시 원본 데이터 사용
        st.warning(f"보간 실패, 원본 데이터 사용: {str(e)}")
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
        luminance_grid = np.zeros_like(theta_grid)
        for i, phi in enumerate(phi_vals):
            for j, theta in enumerate(theta_vals):
                mask = (df_long['Phi'] == phi) & (df_long['Theta'] == theta)
                if mask.any():
                    luminance_grid[i, j] = df_long.loc[mask, 'Luminance'].iloc[0]
        luminance_interp = luminance_grid.T
    
    # Plotly 컬러맵 설정
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
    
    # Plotly figure 생성
    fig = go.Figure()
    
    # 컨투어 플롯 추가
    fig.add_trace(go.Scatterpolar(
        r=theta_grid.flatten(),
        theta=phi_grid.flatten(),
        mode='markers',
        marker=dict(
            size=3,
            color=luminance_interp.flatten(),
            colorscale=plotly_cmap,
            cmin=vmin,
            cmax=vmax,
            showscale=True,
            colorbar=dict(
                title=dict(text="Luminance", side="right"),
                tickmode="linear",
                tick0=vmin,
                dtick=(vmax-vmin)/5
            )
        ),
        name='ISO Data',
        hovertemplate='Theta: %{r:.1f}°<br>Phi: %{theta:.1f}°<br>Luminance: %{marker.color:.2f}<extra></extra>'
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=f'ISO Luminance Distribution (Polar)<br>(Range: {vmin:.1f} - {vmax:.1f})',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, theta_vals.max()],
                tickmode='array',
                tickvals=theta_vals,
                ticktext=[f'{int(t)}°' for t in theta_vals]
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 360, 30)),
                ticktext=[f'{i}°' for i in range(0, 360, 30)],
                direction='clockwise',
                rotation=90
            )
        ),
        width=700,
        height=700,
        font=dict(size=12)
    )
    
    return fig

def create_plotly_cartesian_plot(df_long, vmin, vmax, cmap='Jet', resolution=300):
    """Plotly로 직교좌표계 플롯 생성"""
    
    # 극좌표를 직교좌표로 변환
    theta_rad = np.radians(df_long['Theta'])
    phi_rad = np.radians(df_long['Phi'])
    r_norm = df_long['Theta'] / df_long['Theta'].max()
    
    x = r_norm * np.cos(phi_rad)
    y = r_norm * np.sin(phi_rad)
    
    # 고해상도 격자 생성
    xi = np.linspace(-1.1, 1.1, resolution)
    yi = np.linspace(-1.1, 1.1, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # 보간
    try:
        rbf = Rbf(x, y, df_long['Luminance'], function='multiquadric', smooth=0.1)
        zi = rbf(xi_grid, yi_grid)
    except Exception as e:
        st.warning(f"RBF 보간 실패, griddata 사용: {str(e)}")
        try:
            zi = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), method='linear')
        except Exception as e2:
            st.warning(f"griddata 보간도 실패, 최근접 이웃 사용: {str(e2)}")
            zi = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), method='nearest')
    
    # 원형 마스크
    mask = xi_grid**2 + yi_grid**2 <= 1.02**2
    zi_masked = np.where(mask, zi, np.nan)
    
    # Plotly 컬러맵 설정
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
    
    # Plotly figure 생성
    fig = go.Figure()
    
    # 히트맵 추가
    fig.add_trace(go.Heatmap(
        x=xi,
        y=yi,
        z=zi_masked,
        colorscale=plotly_cmap,
        zmin=vmin,
        zmax=vmax,
        showscale=True,
        colorbar=dict(
            title=dict(text="Luminance", side="right")
        ),
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Luminance: %{z:.2f}<extra></extra>'
    ))
    
    # 원형 경계 추가
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta_circle)
    y_circle = np.sin(theta_circle)
    
    fig.add_trace(go.Scatter(
        x=x_circle,
        y=y_circle,
        mode='lines',
        line=dict(color='white', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 각도 라벨 추가
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in angles:
        x_pos = 1.15 * np.cos(np.radians(angle))
        y_pos = 1.15 * np.sin(np.radians(angle))
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=f'{angle}°',
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
    
    # 레이아웃 설정
    fig.update_layout(
        title=f'ISO Luminance Distribution (Cartesian)<br>(Range: {vmin:.1f} - {vmax:.1f})',
        xaxis=dict(
            range=[-1.3, 1.3],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            range=[-1.3, 1.3], 
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        width=700,
        height=700,
        plot_bgcolor='black',
        font=dict(size=12)
    )
    
    return fig

def create_cross_section_plots(df_long, cross_type, cross_value):
    """크로스섹션 플롯 생성 (matplotlib로 PNG 저장용)"""
    
    try:
        if cross_type == "Theta 고정":
            # 특정 Theta 값에서 Phi에 따른 변화
            df_section = df_long[df_long['Theta'] == cross_value].copy()
            if df_section.empty:
                # 가장 가까운 값 찾기
                closest_theta = df_long['Theta'].iloc[(df_long['Theta'] - cross_value).abs().argsort()[:1]].values[0]
                df_section = df_long[df_long['Theta'] == closest_theta].copy()
                st.info(f"정확한 Theta={cross_value}° 데이터가 없어 가장 가까운 {closest_theta}° 사용")
                
            df_section = df_section.sort_values('Phi')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 선형 플롯
            ax1.plot(df_section['Phi'], df_section['Luminance'], 'b-o', linewidth=2, markersize=4)
            ax1.set_xlabel('Phi (degrees)')
            ax1.set_ylabel('Luminance')
            ax1.set_title(f'Cross-section at Theta = {cross_value}°')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 360)
            
            # 극좌표 플롯
            phi_rad = np.radians(df_section['Phi'])
            ax2 = plt.subplot(122, projection='polar')
            ax2.plot(phi_rad, df_section['Luminance'], 'r-o', linewidth=2, markersize=4)
            ax2.set_title(f'Polar view: Theta = {cross_value}°')
            ax2.set_theta_zero_location('E')
            ax2.set_theta_direction(1)
            
        else:  # Phi 고정
            # 특정 Phi 값에서 Theta에 따른 변화
            df_section = df_long[df_long['Phi'] == cross_value].copy()
            if df_section.empty:
                # 가장 가까운 값 찾기
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - cross_value).abs().argsort()[:1]].values[0]
                df_section = df_long[df_long['Phi'] == closest_phi].copy()
                st.info(f"정확한 Phi={cross_value}° 데이터가 없어 가장 가까운 {closest_phi}° 사용")
                
            df_section = df_section.sort_values('Theta')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 선형 플롯
            ax1.plot(df_section['Theta'], df_section['Luminance'], 'g-o', linewidth=2, markersize=4)
            ax1.set_xlabel('Theta (degrees)')
            ax1.set_ylabel('Luminance')
            ax1.set_title(f'Cross-section at Phi = {cross_value}°')
            ax1.grid(True, alpha=0.3)
            
            # 반지름 방향 플롯 (극좌표의 반지름 축)
            ax2.plot(df_section['Theta'], df_section['Luminance'], 'm-o', linewidth=2, markersize=4)
            ax2.set_xlabel('Theta (degrees) - Radial direction')
            ax2.set_ylabel('Luminance')
            ax2.set_title(f'Radial profile at Phi = {cross_value}°')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"크로스섹션 플롯 생성 중 오류: {str(e)}")
        # 빈 figure 반환
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"크로스섹션 생성 실패\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Error in Cross-section Generation")
        return fig

def create_plotly_cross_section(df_long, cross_type, cross_value):
    """Plotly로 크로스섹션 플롯 생성"""
    
    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'Cross-section at {cross_type} = {cross_value}°', 'Polar View'],
            specs=[[{"secondary_y": False}, {"type": "polar"}]]
        )
        
        if cross_type == "Theta":
            # 특정 Theta 값에서 Phi에 따른 변화
            df_section = df_long[df_long['Theta'] == cross_value].copy()
            if df_section.empty:
                closest_theta = df_long['Theta'].iloc[(df_long['Theta'] - cross_value).abs().argsort()[:1]].values[0]
                df_section = df_long[df_long['Theta'] == closest_theta].copy()
                st.info(f"정확한 Theta={cross_value}° 데이터가 없어 가장 가까운 {closest_theta}° 사용")
                
            df_section = df_section.sort_values('Phi')
            
            # 선형 플롯
            fig.add_trace(
                go.Scatter(x=df_section['Phi'], y=df_section['Luminance'], 
                          mode='lines+markers', name='Luminance',
                          line=dict(color='blue', width=2),
                          hovertemplate='Phi: %{x}°<br>Luminance: %{y:.2f}<extra></extra>'),
                row=1, col=1
            )
            
            # 극좌표 플롯
            fig.add_trace(
                go.Scatterpolar(r=df_section['Luminance'], theta=df_section['Phi'],
                              mode='lines+markers', name='Polar',
                              line=dict(color='red', width=2),
                              hovertemplate='Phi: %{theta}°<br>Luminance: %{r:.2f}<extra></extra>'),
                row=1, col=2
            )
            
        else:  # Phi 고정
            df_section = df_long[df_long['Phi'] == cross_value].copy()
            if df_section.empty:
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - cross_value).abs().argsort()[:1]].values[0]
                df_section = df_long[df_long['Phi'] == closest_phi].copy()
                st.info(f"정확한 Phi={cross_value}° 데이터가 없어 가장 가까운 {closest_phi}° 사용")
                
            df_section = df_section.sort_values('Theta')
            
            # 선형 플롯
            fig.add_trace(
                go.Scatter(x=df_section['Theta'], y=df_section['Luminance'],
                          mode='lines+markers', name='Luminance',
                          line=dict(color='green', width=2),
                          hovertemplate='Theta: %{x}°<br>Luminance: %{y:.2f}<extra></extra>'),
                row=1, col=1
            )
            
            # 반지름 방향 플롯
            fig.add_trace(
                go.Scatter(x=df_section['Theta'], y=df_section['Luminance'],
                          mode='lines+markers', name='Radial',
                          line=dict(color='magenta', width=2),
                          hovertemplate='Theta: %{x}°<br>Luminance: %{y:.2f}<extra></extra>'),
                row=1, col=2
            )
        
        # 레이아웃 업데이트
        fig.update_xaxes(title_text="Phi (degrees)" if cross_type == "Theta" else "Theta (degrees)", row=1, col=1)
        fig.update_yaxes(title_text="Luminance", row=1, col=1)
        
        if cross_type != "Theta":
            fig.update_xaxes(title_text="Theta (degrees)", row=1, col=2)
            fig.update_yaxes(title_text="Luminance", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True, title_text=f"Cross-section Analysis: {cross_type} = {cross_value}°")
        
        return fig
        
    except Exception as e:
        st.error(f"Plotly 크로스섹션 생성 중 오류: {str(e)}")
        # 에러 메시지가 포함된 빈 figure 반환
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"크로스섹션 생성 실패<br>{str(e)}",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Error in Cross-section Generation",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig

def save_plotly_as_html(fig, filename):
    """Plotly 그래프를 HTML로 저장"""
    try:
        html_str = fig.to_html(include_plotlyjs='cdn')
        return html_str.encode()
    except Exception as e:
        st.error(f"HTML 저장 실패: {str(e)}")
        # 기본 HTML 반환
        error_html = f"""
        <html>
        <head><title>Error</title></head>
        <body>
        <h1>Plot Generation Error</h1>
        <p>Error: {str(e)}</p>
        </body>
        </html>
        """
        return error_html.encode()

def save_matplotlib_as_png(fig):
    """Matplotlib 그래프를 PNG 바이트로 저장"""
    try:
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        return img_buffer.getvalue()
    except Exception as e:
        st.error(f"PNG 저장 실패: {str(e)}")
        # 빈 이미지 버퍼 반환
        img_buffer = io.BytesIO()
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Image Generation Error\n{str(e)}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Error")
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        img_buffer.seek(0)
        return img_buffer.getvalue()

def main():
    st.title("📊 Enhanced ISO Polar Plot Visualization")
    st.markdown("CSV 파일을 업로드하여 ISO(광학 강도 분포) polar plot과 크로스섹션을 생성합니다.")

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

            # 컬러맵 선택
            colormap_options = ['Jet', 'Viridis', 'Plasma', 'Inferno', 'Hot', 'Cool', 'Rainbow', 'Turbo']
            selected_cmap = st.sidebar.selectbox(
                "컬러맵",
                colormap_options,
                index=0,
                help="컬러바의 색상 스타일을 선택합니다"
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

            st.sidebar.divider()

            # 크로스섹션 설정
            st.sidebar.subheader("✂️ 크로스섹션 설정")
            
            cross_type = st.sidebar.selectbox(
                "크로스섹션 타입",
                ["Theta 고정", "Phi 고정"],
                help="Theta 고정: 특정 반지름에서 각도별 변화, Phi 고정: 특정 각도에서 반지름별 변화"
            )
            
            if cross_type == "Theta 고정":
                available_values = sorted(df_long['Theta'].unique())
                cross_value = st.sidebar.selectbox("Theta 값 선택", available_values)
            else:
                available_values = sorted(df_long['Phi'].unique())
                cross_value = st.sidebar.selectbox("Phi 값 선택", available_values)

            # 메인 컨텐츠
            tab1, tab2, tab3 = st.tabs(["📊 Main Plots", "✂️ Cross-sections", "📊 Data Info"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Plotly Polar Plot")
                    try:
                        fig_polar = create_plotly_smooth_polar_plot(df_long, vmin, vmax, selected_cmap, resolution)
                        st.plotly_chart(fig_polar, use_container_width=True)
                        
                        # HTML 저장 버튼
                        html_data = save_plotly_as_html(fig_polar, "polar_plot.html")
                        st.download_button(
                            label="🌐 Polar Plot HTML 다운로드",
                            data=html_data,
                            file_name="iso_polar_plot.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.error(f"Polar plot 생성 실패: {str(e)}")
                        st.info("데이터나 설정을 확인해주세요.")
                
                with col2:
                    st.subheader("Plotly Cartesian Plot")
                    try:
                        fig_cartesian = create_plotly_cartesian_plot(df_long, vmin, vmax, selected_cmap, resolution)
                        st.plotly_chart(fig_cartesian, use_container_width=True)
                        
                        # HTML 저장 버튼
                        html_data_cart = save_plotly_as_html(fig_cartesian, "cartesian_plot.html")
                        st.download_button(
                            label="🌐 Cartesian Plot HTML 다운로드",
                            data=html_data_cart,
                            file_name="iso_cartesian_plot.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.error(f"Cartesian plot 생성 실패: {str(e)}")
                        st.info("데이터나 설정을 확인해주세요.")

            with tab2:
                st.subheader(f"크로스섹션: {cross_type} = {cross_value}°")
                
                # Plotly 크로스섹션 (인터랙티브)
                try:
                    fig_cross_plotly = create_plotly_cross_section(df_long, cross_type.split()[0], cross_value)
                    st.plotly_chart(fig_cross_plotly, use_container_width=True)
                    
                    # HTML 저장 버튼
                    html_data_cross = save_plotly_as_html(fig_cross_plotly, "cross_section.html")
                    st.download_button(
                        label="🌐 Cross-section HTML 다운로드",
                        data=html_data_cross,
                        file_name="cross_section.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"인터랙티브 크로스섹션 생성 실패: {str(e)}")
                
                st.divider()
                
                # Matplotlib 크로스섹션 (PNG 저장용)
                st.subheader("크로스섹션 (PNG 저장용)")
                try:
                    fig_cross_mpl = create_cross_section_plots(df_long, cross_type, cross_value)
                    st.pyplot(fig_cross_mpl)
                    
                    # PNG 저장 버튼
                    png_data = save_matplotlib_as_png(fig_cross_mpl)
                    st.download_button(
                        label="🖼️ 크로스섹션 PNG 다운로드",
                        data=png_data,
                        file_name=f"cross_section_{cross_type.replace(' ', '_')}_{cross_value}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"PNG 크로스섹션 생성 실패: {str(e)}")
                    st.info("다른 크로스섹션 값을 시도해보세요.")

            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("데이터 미리보기")
                    st.dataframe(df_long.head(10))
                
                with col2:
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
                
                # 전체 데이터 다운로드
                csv_data = df_long.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 전처리된 데이터 다운로드 (CSV)",
                    data=csv_data,
                    file_name="processed_iso_data.csv",
                    mime="text/csv"
                )

    else:
        # 샘플 데이터 및 사용법 안내
        st.info("👆 사이드바에서 CSV 파일을 업로드하세요.")

        st.subheader("📋 사용법")
        st.markdown("""
        ### 🚀 새로운 기능들:
        
        #### 1. **Plotly 기반 시각화**
        - 모든 주요 플롯이 Plotly로 구현되어 인터랙티브 기능 제공
        - 확대/축소, 패닝, 호버 정보 등 지원
        
        #### 2. **HTML 저장 기능**
        - 모든 Plotly 그래프를 HTML 파일로 저장 가능
        - 브라우저에서 독립적으로 실행 가능한 인터랙티브 파일
        
        #### 3. **크로스섹션 분석**
        - **Theta 고정**: 특정 반지름에서 각도별 변화 분석
        - **Phi 고정**: 특정 각도에서 반지름별 변화 분석
        - Plotly (HTML) + Matplotlib (PNG) 두 가지 형태로 저장
        
        #### 4. **다양한 내보내기 옵션**
        - 📊 메인 플롯: HTML 형태로 저장
        - ✂️ 크로스섹션: HTML (인터랙티브) + PNG (고품질 이미지)
        - 📊 데이터: 전처리된 CSV 파일
        
        ### 📁 **CSV 파일 형식**: 
        - 첫 번째 컬럼: `Theta` (각도 값)
        - 나머지 컬럼: 각 Phi 각도 (0, 10, 20, ..., 360)
        """)

if __name__ == "__main__":
    main()