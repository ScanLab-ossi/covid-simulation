import streamlit as st


def streamlit_theme():
    font = "IBM Plex Mono"
    primary_color = "#F63366"
    font_color = "#262730"
    grey_color = "#f0f2f6"
    base_size = 16
    lg_font = base_size * 1.25
    sm_font = base_size * 0.8  # st.table size
    xl_font = base_size * 1.75

    config = {
        "config": {
            "arc": {"fill": primary_color},
            "area": {"fill": primary_color},
            "circle": {"fill": primary_color, "stroke": font_color, "strokeWidth": 0.5},
            "line": {"stroke": primary_color},
            "path": {"stroke": primary_color},
            "point": {"stroke": primary_color},
            "rect": {"fill": primary_color},
            "shape": {"stroke": primary_color},
            "symbol": {"fill": primary_color},
            "title": {
                "font": font,
                "color": font_color,
                "fontSize": lg_font,
                "anchor": "start",
            },
            "axis": {
                "titleFont": font,
                "titleColor": font_color,
                "titleFontSize": sm_font,
                "labelFont": font,
                "labelColor": font_color,
                "labelFontSize": sm_font,
                "gridColor": grey_color,
                "domainColor": font_color,
                "tickColor": "#fff",
            },
            "header": {
                "labelFont": font,
                "titleFont": font,
                "labelFontSize": base_size,
                "titleFontSize": base_size,
            },
            "legend": {
                "titleFont": font,
                "titleColor": font_color,
                "titleFontSize": sm_font,
                "labelFont": font,
                "labelColor": font_color,
                "labelFontSize": sm_font,
            },
            "range": {
                "category": ["#f63366", "#fffd80", "#0068c9", "#ff2b2b", "#09ab3b"],
                "diverging": [
                    "#850018",
                    "#cd1549",
                    "#f6618d",
                    "#fbafc4",
                    "#f5f5f5",
                    "#93c5fe",
                    "#5091e6",
                    "#1d5ebd",
                    "#002f84",
                ],
                "heatmap": [
                    "#ffb5d4",
                    "#ff97b8",
                    "#ff7499",
                    "#fc4c78",
                    "#ec245f",
                    "#d2004b",
                    "#b10034",
                    "#91001f",
                    "#720008",
                ],
                "ramp": [
                    "#ffb5d4",
                    "#ff97b8",
                    "#ff7499",
                    "#fc4c78",
                    "#ec245f",
                    "#d2004b",
                    "#b10034",
                    "#91001f",
                    "#720008",
                ],
                "ordinal": [
                    "#ffb5d4",
                    "#ff97b8",
                    "#ff7499",
                    "#fc4c78",
                    "#ec245f",
                    "#d2004b",
                    "#b10034",
                    "#91001f",
                    "#720008",
                ],
            },
        }
    }
    return config


def remove_label_css(order):
    order += 5
    # st.markdown(
    #     f"""
    # <style>
    #     section.main > div > div:nth-child(1) > div:nth-child({order}) > div > label {{
    #         display: none;
    #     }}
    # </style>
    # """,
    #     unsafe_allow_html=True,
    # )


def nav_css():
    container = "#root > div:nth-child(1) > div > div > div > div > section.main > div > div:nth-child(1)"
    css = f"""
    <style>
        div[radiogroup] {{
            background-color: red;
        }}
        .stRadio div {{
            display: flex;
            flex-direction: row;
        }}
        .stRadio div label {{
            padding-right: 50px;
        }}
    </style>"""
    st.markdown(css, unsafe_allow_html=True)


def grid_css(repeat):
    container = "#root > div:nth-child(1) > div > div > div > div > section.main > div > div:nth-child(1)"
    css = f"""
        /*{container} {{
            display: grid;
            grid-template-columns: 300px 300px;
            grid-template-rows: repeat({repeat // 2 + 2 + 3}, auto);
            grid-auto-flow: column;
            column-gap: 20px;
        }}
        {container} > div:nth-child(-n+4) {{
            grid-column: 1 / span 2;
        }}
        {container} > div:nth-child({repeat+5}) {{
            grid-column: 1 / span 2;
            # grid-row: {repeat-2};
        }}
        .element-container {{
            width: 100% !important;
        }}

        .Widget {{
            width: 100% !important;
        }}*/
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    print(repeat)


# root > div:nth-child(1) > div > div > div > div > section.main > div > div:nth-child(1) > div:nth-child(19) > div > label

