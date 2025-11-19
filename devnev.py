import streamlit as st
import requests
from PIL import Image
import io
import base64
"""Alternative version for local images"""
    
st.title("About Our Team - DevNev")

# Team data structure
team_data = {
    "Leadership": [
        {"name": "Mme. Rachana SOPHON", "role": "Founder", "image_path": "images/devnev-founder.jpg  "},
    ],
    "Managers": [
        {"name": "Mme. Vanna NEANG (VanneBeUrs)", "role": "Geeneral Manager", "image_path": "images/vannabeurs.jpg"},
        {"name": "Mr. Ly ZueThean", "role": "One-sider Lover Team Leader", "image_path": "images/zuethean.jpg"}
    ]
}

for level, members in team_data.items():
    st.subheader(level)
    
    cols = st.columns(len(members))
    
    for idx, member in enumerate(members):
        with cols[idx]:
            try:
                # Load and display local image
                image = Image.open(member["image_path"])
                st.image(image, width=150)
                st.write(f"**{member['name']}**")
                st.write(f"*{member['role']}*")
            except FileNotFoundError:
                st.error(f"Image not found: {member['image_path']}")
                st.write(f"**{member['name']}**")
                st.write(f"*{member['role']}*")