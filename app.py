import os
from typing import Optional

import pandas as pd
import streamlit as st
import altair as alt
import pycountry


st.set_page_config(page_title="Climate & Energy Data Explorer", layout="wide")


@st.cache_data
def load_csv_from_repo(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to read CSV from repo path: {e}")
            return None
    return None


@st.cache_data
def load_csv_from_upload(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        return None


@st.cache_data
def load_country_codes() -> pd.DataFrame:
    """Load country ISO codes mapping"""
    try:
        return pd.read_csv("country_codes.csv")
    except Exception as e:
        st.error(f"Failed to load country codes: {e}")
        return pd.DataFrame(columns=['country', 'alpha-3'])


def main():
    st.title("Climate & Energy — Data Explorer")

    st.markdown(
        "This interactive dashboard helps you explore the `global-data-on-sustainable-energy (1).csv` dataset. Use the sidebar to filter, pick metrics, and view summary cards and charts."
    )

    default_csv = "global-data-on-sustainable-energy (1).csv"
    df = load_csv_from_repo(default_csv)

    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])  # user can override
    if uploaded is not None:
        df_uploaded = load_csv_from_upload(uploaded)
        if df_uploaded is not None:
            df = df_uploaded

    if df is None:
        st.info(
            "No CSV found in repo and no file uploaded. Add `global-data-on-sustainable-energy (1).csv` to the repo root or upload one here."
        )
        st.stop()

    # heuristics for entity/year/population columns
    entity_candidates = [c for c in df.columns if c.lower() in ("entity", "country", "country name", "region", "location")]
    entity_col = entity_candidates[0] if entity_candidates else None

    pop_candidates = [c for c in df.columns if c.lower() in ("population", "pop", "population_total")]
    pop_col = pop_candidates[0] if pop_candidates else None

    year_like = None
    for candidate in ["Year", "year", "YEAR", "year_id"]:
        if candidate in df.columns:
            year_like = candidate
            break

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Load country codes for choropleth
    country_codes = load_country_codes()
    
    # Global controls in sidebar (applies across tabs)
    st.sidebar.header("Global filters & controls")
    if entity_col:
        all_entities = sorted(df[entity_col].dropna().unique().tolist())
        selected_entities = st.sidebar.multiselect("Entities / Countries", all_entities, default=all_entities[:3])
    else:
        selected_entities = None

    if year_like and pd.api.types.is_numeric_dtype(df[year_like]):
        ymin = int(df[year_like].min())
        ymax = int(df[year_like].max())
        year_range = st.sidebar.slider("Year range", ymin, ymax, (ymin, ymax))
    else:
        year_range = None

    metric_choice = st.sidebar.selectbox("Metric (numeric)", numeric_cols) if numeric_cols else None

    st.sidebar.markdown("---")
    st.sidebar.markdown("Tip: use the Compare tab for multi-country small multiples and normalization options.")

    # Apply base filters
    df_filtered = df.copy()
    if selected_entities and entity_col:
        df_filtered = df_filtered[df_filtered[entity_col].isin(selected_entities)]
    if year_range and year_like:
        df_filtered = df_filtered[(df_filtered[year_like] >= year_range[0]) & (df_filtered[year_like] <= year_range[1])]

    # Tabs: Overview | Explore | Compare | Export
    tab_overview, tab_explore, tab_compare, tab_export = st.tabs(["Overview", "Explore", "Compare", "Export"])

    # ---------- Overview tab ----------
    with tab_overview:
        st.header("Overview")
        # KPI cards with small sparklines
        c1, c2, c3, c4 = st.columns(4)

        # Rows
        c1.metric("Rows (filtered)", f"{df_filtered.shape[0]}")
        # sparkline for rows over time (counts per year)
        if year_like and year_like in df.columns:
            # avoid reset_index collisions by grouping with as_index=False
            rows_ts = df.groupby(year_like, as_index=False).agg(count=(year_like, "size"))
            spark = alt.Chart(rows_ts).mark_line(interpolate='monotone').encode(x=year_like, y='count')
            c1.altair_chart(spark.properties(height=60), use_container_width=True)

        # Metric mean + sparkline
        if metric_choice:
            mean_val = df_filtered[metric_choice].mean()
            c2.metric(f"Mean {metric_choice}", f"{mean_val:.3g}")
            if year_like and year_like in df.columns:
                # group with as_index=False to avoid creating duplicate column names when resetting index
                m_ts = df_filtered.groupby(year_like, as_index=False)[metric_choice].mean()
                spark2 = alt.Chart(m_ts).mark_line(interpolate='monotone').encode(x=year_like, y=metric_choice)
                c2.altair_chart(spark2.properties(height=60), use_container_width=True)
        else:
            c2.write("\n")

        # Latest year metric
        if metric_choice and year_like and year_like in df_filtered.columns:
            latest_y = int(df_filtered[year_like].max())
            latest_mean = df_filtered[df_filtered[year_like] == latest_y][metric_choice].mean()
            c3.metric(f"{metric_choice} ({latest_y})", f"{latest_mean:.3g}")
            # sparkline for latest years (last 5)
            if year_like:
                last_n = df_filtered.groupby(year_like, as_index=False)[metric_choice].mean().sort_values(by=year_like).tail(5)
                if not last_n.empty:
                    spark3 = alt.Chart(last_n).mark_line().encode(x=year_like, y=metric_choice)
                    c3.altair_chart(spark3.properties(height=60), use_container_width=True)
        else:
            c3.write("\n")

        # Distinct entities
        if entity_col:
            c4.metric("Distinct entities", f"{df_filtered[entity_col].nunique()}")
        else:
            c4.write("\n")

        st.markdown("---")
        st.subheader("Snapshot of filtered data")
        st.dataframe(df_filtered.head(200))

    # ---------- Explore tab ----------
    with tab_explore:
        st.header("Explore")
        st.markdown("Use filters from the sidebar to shape the dataset. Choose columns to view and chart options below.")

        # column picker and chart settings
        cols = list(df_filtered.columns)
        selected_cols = st.multiselect("Columns to display", cols, default=cols[:min(10, len(cols))])
        st.dataframe(df_filtered[selected_cols])

        st.markdown("---")
        st.subheader("Chart")
        if metric_choice is None:
            st.info("No numeric metric detected. Upload or choose a dataset with numeric columns.")
        else:
            chart_type = st.selectbox("Chart type", ["line", "area", "bar"], index=0)
            chart_df = df_filtered.dropna(subset=[metric_choice])
            if year_like and year_like in chart_df.columns:
                x_enc = alt.X(year_like, title=year_like)
            else:
                chart_df = chart_df.reset_index()
                x_enc = alt.X('index:Q', title='index')

            base = alt.Chart(chart_df).encode(x=x_enc, y=alt.Y(metric_choice, title=metric_choice), tooltip=[metric_choice, entity_col, year_like])
            if chart_type == 'line':
                st.altair_chart(base.mark_line(point=True).interactive().properties(height=450), use_container_width=True)
            elif chart_type == 'area':
                st.altair_chart(base.mark_area(opacity=0.4).interactive().properties(height=450), use_container_width=True)
            else:
                st.altair_chart(base.mark_bar().interactive().properties(height=450), use_container_width=True)

        # Additional visualizations: correlation heatmap and energy mix (stacked area)
        st.markdown("---")

        # Chart presets
        st.subheader("Chart presets")
        preset_options = {
            "None": None,
            "Renewables share": ["Electricity from renewables (TWh)", "Renewables (% electricity)", "Renewable energy share in total final energy consumption (%)"],
            "CO2 vs GDP": ["Annual CO2 emissions (tonnes)", "GDP per capita", "CO2 emissions per capita"],
            "Energy access": ["Access to electricity (% of population)", "Access to clean fuels for cooking"],
            "Energy intensity": ["Energy intensity level of primary energy (MJ/$2017 PPP GDP)", "Primary energy consumption per capita (kWh/person)"]
        }
        selected_preset = st.selectbox("Quick view presets", list(preset_options.keys()))
        if selected_preset != "None" and preset_options[selected_preset]:
            available_preset_cols = [col for col in preset_options[selected_preset] if col in df_filtered.columns]
            if available_preset_cols:
                preset_df = df_filtered[available_preset_cols + [year_like] + ([entity_col] if entity_col else [])].dropna()
                preset_long = preset_df.melt(id_vars=[year_like, entity_col] if entity_col else [year_like], 
                                          value_vars=available_preset_cols, 
                                          var_name='metric', 
                                          value_name='value')
                preset_chart = alt.Chart(preset_long).mark_line(point=True).encode(
                    x=alt.X(year_like + ':Q', title=year_like),
                    y=alt.Y('value:Q', title='Value'),
                    color='metric:N',
                    strokeDash='metric:N',
                    tooltip=[year_like, 'metric', 'value', entity_col] if entity_col else [year_like, 'metric', 'value']
                ).properties(height=400)
                st.altair_chart(preset_chart.interactive(), use_container_width=True)
            else:
                st.warning(f"None of the columns for the {selected_preset} preset are available in the current dataset")

        st.markdown("---")
        st.subheader("Correlation heatmap (numeric columns)")
        num_df = df_filtered.select_dtypes(include=["number"]).dropna(axis=1, how='all')
        if num_df.shape[1] < 2:
            st.info("Not enough numeric columns to compute correlations.")
        else:
            corr = num_df.corr()
            # prepare for Altair heatmap
            corr_reset = corr.reset_index().melt(id_vars=corr.index.name or 'index')
            # column names for encoding
            x_col = 'variable'
            y_col = corr.index.name or 'index'
            val_col = 'value'
            # If melt produced different names, normalize them
            if 'index' not in corr_reset.columns and corr.index.name:
                corr_reset = corr.reset_index().melt(id_vars=corr.index.name)

            heat = alt.Chart(corr_reset).mark_rect().encode(
                x=alt.X(x_col + ':N', sort=None, title=None),
                y=alt.Y(y_col + ':N', sort=None, title=None),
                color=alt.Color(val_col + ':Q', scale=alt.Scale(scheme='redblue')),
                tooltip=[x_col + ':N', y_col + ':N', val_col + ':Q']
            ).properties(height=500)
            text = heat.mark_text(baseline='middle', fontSize=11).encode(text=alt.Text(val_col + ':Q', format='.2f'))
            st.altair_chart((heat + text).configure_axis(labelAngle=0), use_container_width=True)

        st.markdown("---")
        st.subheader("Energy mix (stacked area)")
        # Manual column selection for energy mix
        st.markdown("Select columns to include in the stacked area chart:")
        # try to detect common energy source columns as suggestions
        energy_terms = ['electricity from fossil', 'electricity from nuclear', 'electricity from renewables', 'renewable', 'fossil', 'nuclear']
        candidates = [c for c in df_filtered.columns if any(t.lower() in c.lower() for t in energy_terms)]
        # prefer exact known columns if present
        preferred = [
            'Electricity from fossil fuels (TWh)',
            'Electricity from nuclear (TWh)',
            'Electricity from renewables (TWh)'
        ]
        suggested_cols = [c for c in preferred if c in df_filtered.columns]
        if not suggested_cols:
            # fall back to any candidate with numeric types
            suggested_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df_filtered[c])]
        
        # Let user select columns from numeric columns
        numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()
        mix_cols = st.multiselect(
            "Columns to include",
            options=numeric_cols,
            default=suggested_cols if suggested_cols else numeric_cols[:3],
            help="Select numeric columns to include in the stacked area chart. Best suited for related metrics that can be meaningfully stacked (e.g., different energy sources)."
        )

        if not mix_cols or (year_like is None or year_like not in df_filtered.columns):
            st.info("Could not find suitable energy source columns and a Year column for a stacked area chart.")
        else:
            mix_df = df_filtered[[year_like] + mix_cols].dropna(subset=[year_like])
            # aggregate per year
            mix_agg = mix_df.groupby(year_like, as_index=False)[mix_cols].sum()
            # transform to long form for stacked area
            mix_long = mix_agg.melt(id_vars=[year_like], value_vars=mix_cols, var_name='source', value_name='value')
            area = alt.Chart(mix_long).mark_area(opacity=0.8).encode(
                x=alt.X(year_like + ':Q', title=year_like),
                y=alt.Y('value:Q', stack='normalize', title='Share (normalized 0-1)'),
                color=alt.Color('source:N', title='Source'),
                tooltip=[year_like + ':Q', 'source:N', 'value:Q']
            ).properties(height=400)
            st.altair_chart(area.interactive(), use_container_width=True)

        # Choropleth map
        st.markdown("---")
        st.subheader("World map view")
        
        if not metric_choice or not entity_col:
            st.info("Select an entity column and metric to view the choropleth map")
        else:
            # For choropleth, we need the latest year's data for each country
            if year_like and year_like in df_filtered.columns:
                latest_year = df_filtered[year_like].max()
                map_data = df_filtered[df_filtered[year_like] == latest_year].copy()
            else:
                map_data = df_filtered.copy()
            
            # Merge with country codes
            map_data = map_data.merge(country_codes, 
                                    left_on=entity_col, 
                                    right_on='country', 
                                    how='left')
            
            # Remove entries without ISO codes
            map_data = map_data.dropna(subset=['alpha-3'])
            
            if map_data.empty:
                st.warning("No mapping data available. Make sure country names match between your data and our ISO codes.")
            else:
                # Create map using Altair
                source = alt.topo_feature('https://raw.githubusercontent.com/vega/vega-datasets/master/data/world-110m.json', 
                                        'countries')
                
                background = alt.Chart(source).mark_geoshape(fill='lightgray')
                
                choropleth = alt.Chart(source).mark_geoshape().encode(
                    color=alt.Color(f'{metric_choice}:Q', 
                                  scale=alt.Scale(scheme='blueorange'),
                                  title=metric_choice),
                    tooltip=[
                        alt.Tooltip('name:N', title='Country'),
                        alt.Tooltip(f'{metric_choice}:Q', title=metric_choice)
                    ]
                ).transform_lookup(
                    lookup='id',
                    from_=alt.LookupData(map_data, 'alpha-3', [metric_choice, entity_col])
                ).project(
                    'equirectangular'
                ).properties(
                    width=700,
                    height=400
                )
                
                st.altair_chart((background + choropleth).configure_view(strokeWidth=0), use_container_width=True)    # ---------- Compare tab ----------
    with tab_compare:
        st.header("Compare — multi-country small multiples")
        st.markdown("Choose multiple entities in the sidebar to compare. Use normalization for fair comparisons.")

        if not entity_col:
            st.info("No entity/country column detected in the dataset. Upload a dataset with a country column to use this view.")
        elif not metric_choice:
            st.info("No numeric metric chosen. Select one in the sidebar.")
        else:
            norm = st.radio("Normalization", ["None", "Normalize 0-1", "Per-capita (population required)"], index=0)

            comp_df = df_filtered.dropna(subset=[metric_choice])
            # apply per-capita if requested and population column exists
            if norm == "Per-capita (population required)":
                if pop_col and pop_col in comp_df.columns:
                    comp_df = comp_df.copy()
                    comp_df[metric_choice + '_percap'] = comp_df[metric_choice] / comp_df[pop_col]
                    use_col = metric_choice + '_percap'
                else:
                    st.warning("No population column found — falling back to unnormalized values.")
                    use_col = metric_choice
            elif norm == "Normalize 0-1":
                comp_df = comp_df.copy()
                # normalize within each entity
                comp_df['normalized'] = comp_df.groupby(entity_col)[metric_choice].transform(lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0)
                use_col = 'normalized'
            else:
                use_col = metric_choice

            # small multiples: facet by entity
            # limit number of entities to avoid huge charts
            entities = selected_entities if selected_entities else sorted(comp_df[entity_col].dropna().unique().tolist())[:6]
            comp_df = comp_df[comp_df[entity_col].isin(entities)]

            if year_like and year_like in comp_df.columns:
                chart = alt.Chart(comp_df).mark_line(point=True).encode(
                    x=alt.X(year_like, title=year_like),
                    y=alt.Y(use_col, title=use_col),
                    color=alt.Color(entity_col + ':N', legend=None),
                    tooltip=[entity_col, year_like, use_col]
                ).properties(width=250, height=120).facet(row=entity_col)
                st.altair_chart(chart, use_container_width=True)
            else:
                # fallback: multiple lines on same chart
                chart = alt.Chart(comp_df).mark_line(point=True).encode(x='index:Q', y=use_col, color=entity_col, tooltip=[entity_col, use_col]).interactive()
                st.altair_chart(chart.properties(height=450), use_container_width=True)

    # ---------- Export tab ----------
    with tab_export:
        st.header("Export")
        st.markdown("Download the current filtered dataset or presets.")
        csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered-data.csv", mime="text/csv")
        if st.button("Download displayed columns as CSV"):
            cols = list(df_filtered.columns)[:min(20, len(df_filtered.columns))]
            st.download_button("Download sample CSV", data=df_filtered[cols].to_csv(index=False).encode('utf-8'), file_name="sample.csv")

    # End of main


if __name__ == "__main__":
    main()
