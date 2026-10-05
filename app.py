"""Streamlit interface for the AI-Based College Academic Assistant."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import streamlit as st

import backend_api


APP_TITLE = "AI-Based College Academic Assistant"
NO_INFO_MESSAGE = "I couldn't find this information in the college documents."
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <style>
    :root {
        --campus-ink: #17313a;
        --campus-teal: #176b68;
        --campus-coral: #c65f43;
        --campus-paper: #f5f7f4;
    }
    .stApp { background: var(--campus-paper); color: var(--campus-ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { border-right: 1px solid #d9e2df; }
    h1, h2, h3 { color: var(--campus-ink); }
    div.stButton > button[kind="primary"] {
        background: var(--campus-teal);
        border-color: var(--campus-teal);
    }
    div.stButton > button[kind="primary"]:hover {
        background: #105654;
        border-color: #105654;
    }
    .assistant-kicker {
        color: var(--campus-coral);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialize_state() -> None:
    """Create browser-session state used by the chat and planner views."""
    st.session_state.setdefault("assistant_session_id", uuid.uuid4().hex)
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("planner_result", None)
    st.session_state.setdefault("planner_inputs", None)
    st.session_state.setdefault("comparison_result", None)


def _render_sources(source_documents: list[dict[str, Any]]) -> None:
    """Render source metadata returned by the backend API."""
    if not source_documents:
        return
    with st.expander(f"Retrieved source documents ({len(source_documents)})"):
        for document in source_documents:
            source = document.get("source", "College document")
            page = document.get("page")
            label = f"{source}, page {page}" if page is not None else str(source)
            st.markdown(f"- {label}")


def _workflow_steps(message: dict[str, Any]) -> list[tuple[str, str]]:
    """Map real graph nodes to visible workflow stages without inventing nodes."""
    nodes = set(message.get("nodes", []))
    answer = message.get("content", "")
    used_rag = "rag" in nodes

    if not used_rag:
        generation_status = "Not run"
    elif answer == NO_INFO_MESSAGE:
        generation_status = "Skipped; no relevant documents found"
    elif "I need an LLM API key" in answer:
        generation_status = "Attempted; LLM configuration is missing"
    else:
        generation_status = "Ran"

    return [
        ("Question analysis", "Ran" if "classify" in nodes else "Not run"),
        ("Retrieval", "Ran" if used_rag else "Not run"),
        ("Generation", generation_status),
        ("Review", "Not implemented in the current workflow"),
    ]


def _render_workflow(message: dict[str, Any]) -> None:
    """Show conceptual stages and the exact LangGraph nodes that executed."""
    steps = _workflow_steps(message)
    with st.expander("Workflow steps"):
        for label, status in steps:
            st.markdown(f"**{label}:** {status}")
        nodes = message.get("nodes", [])
        if nodes:
            st.caption("Graph nodes: " + " -> ".join(nodes))


def _render_chat_message(message: dict[str, Any]) -> None:
    """Render one saved chat turn, including citations and graph trace."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("source_documents", []))
            _render_workflow(message)


def _render_chat_page() -> None:
    """Render the chat assistant and its session-scoped conversation history."""
    st.markdown('<div class="assistant-kicker">Academic support</div>', unsafe_allow_html=True)
    st.header("Chat Assistant")
    st.caption("Ask about college documents, study plans, CGPA, and academic events.")

    for message in st.session_state.chat_messages:
        _render_chat_message(message)

    query = st.chat_input("Ask a question")
    if not query:
        return

    st.session_state.chat_messages.append({"role": "user", "content": query})
    _render_chat_message(st.session_state.chat_messages[-1])

    with st.chat_message("assistant"):
        with st.spinner("Working through your question..."):
            result = backend_api.ask_question(
                query.strip(),
                st.session_state.assistant_session_id,
            )
        response = {
            "role": "assistant",
            "content": result.get("answer", "I couldn't complete that request."),
            "source_documents": result.get("source_documents", []),
            "nodes": result.get("nodes", []),
        }
        st.markdown(response["content"])
        _render_sources(response["source_documents"])
        _render_workflow(response)
    st.session_state.chat_messages.append(response)


def _plan_table_rows(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten day-level planner entries into rows suitable for a data table."""
    rows = []
    for day in plan:
        for allocation in day.get("subjects", []):
            rows.append({
                "Date": day.get("date"),
                "Subject": allocation.get("subject"),
                "Study hours": allocation.get("hours"),
                "Exam date": allocation.get("exam_date"),
                "Daily total": day.get("total_hours"),
            })
    return rows


def _show_plan_result(result: dict[str, Any] | None) -> None:
    """Display plan status, warnings, and its daily allocations."""
    if not result:
        return
    status = result.get("status", "error")
    if status == "error":
        st.error(result.get("message", "Could not create the study plan."))
        return
    if status == "warning":
        st.warning(result.get("message", "The plan was created with warnings."))
    else:
        st.success(result.get("message", "Study plan created."))

    for warning in result.get("warnings", []):
        st.caption("Warning: " + warning)
    rows = _plan_table_rows(result.get("plan", []))
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No study sessions could be scheduled for these dates.")


def _render_planner_page() -> None:
    """Render study-plan creation and modification controls."""
    st.markdown('<div class="assistant-kicker">Plan your preparation</div>', unsafe_allow_html=True)
    st.header("Study Planner")
    st.caption("Create a schedule, then refine it with a plain-language instruction.")

    with st.form("study_plan_form"):
        subject_text = st.text_input(
            "Subjects",
            placeholder="Mathematics, Physics, Chemistry",
            help="Enter multiple subjects separated by commas.",
        )
        col_hours, col_date = st.columns(2)
        with col_hours:
            hours_per_day = st.number_input(
                "Study hours per day",
                min_value=0.5,
                max_value=24.0,
                value=4.0,
                step=0.5,
            )
        with col_date:
            exam_date = st.date_input("Exam date", value=date.today())
        generate = st.form_submit_button("Generate Plan", type="primary")

    if generate:
        subjects = [subject.strip() for subject in subject_text.split(",") if subject.strip()]
        if not subjects:
            st.error("Enter at least one subject.")
        else:
            with st.spinner("Building your study plan..."):
                result = backend_api.create_study_plan(
                    subjects,
                    float(hours_per_day),
                    exam_date.isoformat(),
                    st.session_state.assistant_session_id,
                )
            st.session_state.planner_result = result
            st.session_state.planner_inputs = {
                "subjects": subjects,
                "hours_per_day": float(hours_per_day),
                "exam_date": exam_date.isoformat(),
            }

    _show_plan_result(st.session_state.planner_result)

    st.subheader("Modify this plan")
    if not st.session_state.planner_inputs:
        st.caption("Generate a plan before requesting changes.")
    with st.form("modify_plan_form"):
        instruction = st.text_input(
            "Change request",
            placeholder="Add more revision for Maths",
            disabled=not bool(st.session_state.planner_inputs),
        )
        update = st.form_submit_button(
            "Update Plan",
            type="primary",
            disabled=not bool(st.session_state.planner_inputs),
        )

    if update:
        if not instruction.strip():
            st.error("Describe the change you want to make.")
        else:
            with st.spinner("Updating your study plan..."):
                result = backend_api.modify_study_plan(
                    instruction,
                    st.session_state.assistant_session_id,
                )
            st.session_state.planner_result = result
            st.rerun()


def _render_comparison_page() -> None:
    """Compare the direct LLM response with the document-grounded RAG answer."""
    st.markdown('<div class="assistant-kicker">Grounded vs. general response</div>', unsafe_allow_html=True)
    st.header("RAG vs Basic LLM")
    st.caption("Compare a direct model response with an answer grounded in college documents.")

    with st.form("comparison_form"):
        query = st.text_input("Question", placeholder="What is the attendance requirement?")
        compare = st.form_submit_button("Compare answers", type="primary")

    if compare:
        if not query.strip():
            st.error("Enter a question to compare.")
        else:
            with st.spinner("Getting both answers..."):
                basic_answer = backend_api.ask_basic_llm(query)
                rag_result = backend_api.ask_question(
                    query,
                    st.session_state.assistant_session_id,
                )
            st.session_state.comparison_result = {
                "basic_answer": basic_answer,
                "rag_result": rag_result,
            }

    result = st.session_state.comparison_result
    if result:
        basic_column, rag_column = st.columns(2, gap="large")
        with basic_column:
            st.subheader("Basic LLM")
            st.markdown(result["basic_answer"])
            st.caption("Direct model response; no document retrieval.")
        with rag_column:
            st.subheader("RAG Answer")
            st.markdown(result["rag_result"].get("answer", ""))
            _render_sources(result["rag_result"].get("source_documents", []))
            st.caption("Retrieved context is shown above when available.")


_initialize_state()

with st.sidebar:
    st.markdown("### Academic Assistant")
    page = st.radio(
        "Pages",
        ["Chat Assistant", "Study Planner", "RAG vs Basic LLM"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        reset_complete = False
        try:
            backend_api.reset_session(st.session_state.assistant_session_id)
            st.session_state.chat_messages = []
            st.session_state.planner_result = None
            st.session_state.planner_inputs = None
            st.session_state.comparison_result = None
            reset_complete = True
        except Exception as exc:
            st.error(f"Could not clear this session: {exc}")
        if reset_complete:
            st.rerun()
    st.caption("You can also ask the assistant to calculate CGPA or manage calendar events.")

st.title(APP_TITLE)
if page == "Chat Assistant":
    _render_chat_page()
elif page == "Study Planner":
    _render_planner_page()
else:
    _render_comparison_page()