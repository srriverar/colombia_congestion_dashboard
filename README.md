# 🇨🇴 Dashboard SIN Nodal — Análisis de Congestión del Sistema Eléctrico Colombiano

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

Plataforma interactiva de análisis de congestión de transmisión en el **Sistema Interconectado Nacional (SIN)** de Colombia.

## 🎯 Características

- **9 hojas interactivas** con más de 95 controles
- **Modelo PyPSA real** de 7 nodos con optimización LOPF (HiGHS)
- **Escenarios climáticos**: El Niño / La Niña / Normal
- **BESS** en 3 ubicaciones (La Guajira, Bogotá, Caribe)
- **Diagrama Sankey** de flujo de energía
- **Calculadora de huella de carbono**
- **Juego de presupuesto** (€3B)
- **Desafíos gamificados** (3 retos interactivos)
- **Analizador costo-beneficio** (VPN, TIR, Payback)
- **12 datasets oficiales** de UPME, EPM, CREG, XM

## 🚀 Instalación Local

```bash
# 1. Clonar repositorio
git clone https://github.com/TU_USUARIO/Colombia_Congestion_Dashboard.git
cd Colombia_Congestion_Dashboard

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

## 📊 Hojas del Dashboard

| # | Hoja | Interactividad |
|---|------|---------------|
| 1 | 🏠 Inicio | Calculadora rápida + 3 desafíos gamificados |
| 2 | 🗺️ Mapa de Red | Filtros voltaje + proyección demanda + what-if |
| 3 | ⚡ Arquitectura SIN | Constructor capacidad + balance regional (18 controles) |
| 4 | 🔴 Corredores | 7 sliders + juego presupuesto €3B + costo-beneficio |
| 5 | 💰 Precios Nodales | PyPSA real + heatmap + spread + demand response + CO₂ |
| 6 | 🎮 Simulador | 17 controles + 6 presets + Sankey + comparación A vs B |
| 7 | 📊 Escenarios | 4 vistas: proyección, viaje temporal, radar, inversión |
| 8 | 🗄️ Datos Colombia | Búsqueda + filtro de 12 datasets oficiales |
| 9 | 📋 Política | 8 políticas + simulador de impacto |

## 🛠️ Tecnologías

- **Python 3.10+**
- **Streamlit** — Framework de UI
- **PyPSA** — Modelado de sistemas de potencia
- **HiGHS** — Solver de optimización lineal
- **Plotly** — Visualizaciones interactivas
- **Grid2Poster** — Visualización de red desde OSM

## 📚 Fuentes de Datos

- [UPME - Plan de Expansión de Transmisión 2025-2039](https://www.minenergia.gov.co/documents/15647/Plan-expansion-2025-2039.pdf)
- [Awesome-Electrical-Grid-Mapping](https://github.com/open-energy-transition/Awesome-Electrical-Grid-Mapping)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [XM - Operador del Sistema](https://www.xm.com.co/)
- [CREG](https://www.creg.gov.co/)

## 👤 Autor

**Prof. Sergio Rivera** — Universidad Nacional de Colombia

## 📄 Licencia

Este proyecto es una plataforma de prueba de concepto con fines educativos y de investigación.
