# """
#     CS5001 Fall 2022
#     Assignment number of info
#     Name / Partner
# """
#
# import streamlit as st
#
#
# class MultiPage:
#     def __init__(self):
#         self.pages = []
#
#     def add_page(self, title, func):
#         self.pages.append({
#             'title': title,
#             'function': func
#         })
#
#     def run(self):
#         # select a page title from sidebar
#         page = st.sidebar.selectbox(
#             "Type of loss function",
#             self.pages,
#             # it receives the option as an argument and its output will be cast to str
#             format_func=lambda page: page['title']
#         )
#         page['function']()  # run the function of the selected page title
