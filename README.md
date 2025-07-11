# 📊 ISO Polar Plot Visualization

A comprehensive web application for visualizing and analyzing ISO (Illuminance Spatial Orientation) polar plots with advanced cross-section analysis capabilities.

## ✨ Features

### 🎯 **Main Visualizations**
- **Polar Plot**: Interactive polar coordinate visualization with clean, label-free design
- **Cartesian Plot**: Circular heatmap representation in Cartesian coordinates
- **Advanced Interpolation**: Gaussian filtering and RBF interpolation for smooth surfaces
- **No Empty Spaces**: Enhanced interpolation eliminates gaps in data visualization

### ✂️ **Cross-Section Analysis**
- **Horizontal Cross-Section (0°-180°)**: Right (0°) and left (180°) luminance profiles in a single graph
- **Vertical Cross-Section (90°-270°)**: Top (90°) and bottom (270°) luminance profiles in a single graph
- **Symmetry Analysis**: Compare opposing directions for pattern analysis
- **Multiple Export Formats**: HTML, PNG, and CSV downloads

### 🎨 **Customization Options**
- Multiple color schemes (Jet, Viridis, Plasma, Inferno, Hot, Cool, Rainbow, Turbo)
- Adjustable resolution (100-500 points)
- Manual or automatic color range settings
- Interactive zoom, pan, and hover features

### 💾 **Export Capabilities**
- **HTML**: Interactive plots for web sharing
- **PNG**: High-resolution images for publications
- **CSV**: Raw data for further analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Edge)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/iso-polar-plot.git
   cd iso-polar-plot
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run iso-plot.py
   ```

5. **Open in browser:**
   Navigate to `http://localhost:8501`

## 📁 Input Data Format

### CSV File Structure
Your CSV file should follow this format:

```csv
Theta,0,10,20,30,...,350,360
0,100.5,98.2,95.1,...,102.3,100.5
5,95.8,93.4,90.2,...,97.1,95.8
10,90.3,88.1,85.5,...,92.4,90.3
...
```

- **First column**: `Theta` (polar angle values in degrees)
- **Remaining columns**: Phi angles (0, 10, 20, ..., 360 degrees)
- **Data values**: Luminance measurements

### Sample Data
The application automatically handles:
- Missing data points (uses nearest neighbor interpolation)
- Irregular angular spacing
- Different theta and phi ranges

## 🖥️ User Interface

### Main Interface
- **Sidebar**: Upload files and configure visualization settings
- **Main Area**: Three tabs for different views
  - 📊 **Main Plots**: Polar and Cartesian visualizations
  - ✂️ **Cross-sections**: Horizontal and vertical luminance profiles
  - 📊 **Data Info**: Statistics and data preview

### Cross-Section Analysis
- **Horizontal Direction**: Compares 0° (right) vs 180° (left)
- **Vertical Direction**: Compares 90° (top) vs 270° (bottom)
- **Single Graph Display**: Both directions shown with different colors/markers
- **Export Options**: HTML (interactive), PNG (publication-ready), CSV (data analysis)

## 🔧 Technical Details

### Interpolation Methods
1. **Regular Grid Interpolation**: Primary method for smooth surfaces
2. **RBF (Radial Basis Function)**: Fallback for irregular data
3. **Gaussian Filtering**: Post-processing for surface smoothing
4. **Nearest Neighbor**: Final fallback for sparse data

### Performance Optimization
- **Caching**: Streamlit caching for data processing
- **Memory Management**: Efficient handling of large datasets
- **Resolution Control**: Adjustable interpolation density

### File Outputs
- **HTML Files**: Self-contained interactive plots
- **PNG Images**: 300 DPI publication-quality images
- **CSV Data**: Processed cross-section data for analysis

## 📊 Cross-Section CSV Format

### Horizontal Cross-Section Output
```csv
Theta,Luminance_0deg,Luminance_180deg
0,100.5,98.2
5,95.8,93.4
10,90.3,88.1
...
```

### Vertical Cross-Section Output
```csv
Theta,Luminance_90deg,Luminance_270deg
0,102.1,99.8
5,97.5,94.2
10,92.8,89.5
...
```

## 🛠️ Development

### Project Structure
```
iso-polar-plot/
├── iso-plot.py          # Main application
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── data/               # Sample data files (optional)
```

### Key Dependencies
- **streamlit**: Web application framework
- **plotly**: Interactive plotting
- **matplotlib**: Static plotting
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scipy**: Scientific computing and interpolation

## 🎯 Use Cases

### Research Applications
- **Optical Engineering**: LED and lighting analysis
- **Photometry**: Illuminance distribution studies
- **Quality Control**: Product testing and validation

### Academic Applications
- **Research Papers**: Publication-ready visualizations
- **Presentations**: Interactive demonstrations
- **Data Analysis**: Quantitative luminance studies

### Industrial Applications
- **Product Development**: Lighting design optimization
- **Standards Compliance**: ISO measurement analysis
- **Performance Testing**: Comparative studies

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Test with various data formats
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Bug Reports & Feature Requests

Please use the [GitHub Issues](https://github.com/yourusername/iso-polar-plot/issues) page to:
- Report bugs
- Request new features
- Ask questions
- Share feedback

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualization powered by [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)
- Scientific computing with [SciPy](https://scipy.org/)

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

⭐ **Star this repository if you find it useful!** ⭐
