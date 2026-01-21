from streamlit_extras.stylable_container import stylable_container


def block(key: str, border: int):
    return stylable_container(
        key,
        css_styles=f"""
        {{
            border: {border}px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.5rem;
            padding: calc(1em - 1px)
        }}
        """,
    )
