"""pages/agent.py — Complete Agentic AI & Chatbot Page"""

import streamlit as st
import pandas as pd
from modules.data_loader import load_and_clean_data
from modules.agent import agent_respond, nl_to_pandas, build_rag_context
from utils.helpers import info_card

EXAMPLE_QUERIES = [
    "What are the top 5 stores by total revenue?",
    "Compare Store 5 and Store 20",
    "Show monthly sales trend for 2011",
    "Holiday vs non-holiday sales comparison",
    "Average sales for Store 10 in 2012",
    "Best performing department",
    "Which store had the highest single-week sales?",
    "Show anomalies for Store 1",
    "What is the total revenue in 2010?",
    "How did fuel prices affect sales?",
    "Show Store 14 summary",
    "Forecast Store 10 sales for 12 weeks",
]


def render():
    st.markdown('<div class="section-header">🤖 Agentic AI Assistant</div>', unsafe_allow_html=True)
    info_card(
        "Agentic Architecture",
        "The agent uses a ReAct (Reason → Act → Observe) loop: "
        "1) Parses intent from your query, "
        "2) Selects the best tool (6 available), "
        "3) Executes the tool to get data, "
        "4) Injects tool output + RAG context into LLaMA 3 via Groq, "
        "5) Returns a synthesised business-grade response. "
        "NL queries work offline (no API key needed).",
        "#3b82f6",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── API key ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔑 Groq API Setup")
        api_key = st.text_input(
            "API Key",
            value=st.session_state.get("groq_api_key", ""),
            type="password",
            placeholder="gsk_...",
            help="Free key at console.groq.com — 30 req/min on free tier",
        )
        if api_key:
            st.session_state["groq_api_key"] = api_key
            st.success("✅ Key saved")
        else:
            st.warning("No key → NL queries work, LLM responses disabled")

        st.markdown("---")
        st.markdown("### 💡 Quick Examples")
        for q in EXAMPLE_QUERIES:
            if st.button(q, key=f"qb_{q[:15]}", use_container_width=True):
                st.session_state["preset_query"] = q
                st.rerun()

        st.markdown("---")
        st.markdown("### 🔧 Tools Available")
        tools = [
            ("🔍", "Sales Query", "NL → Pandas"),
            ("🏪", "Store Summary", "KPIs for one store"),
            ("⚖️", "Store Compare", "Side-by-side analysis"),
            ("📈", "Trend Analysis", "Month-over-month"),
            ("🚨", "Anomaly Report", "Flag unusual weeks"),
            ("🔮", "Forecast Summary", "ARIMA/SARIMA status"),
        ]
        for icon, name, desc in tools:
            st.markdown(f"**{icon} {name}** — {desc}")

    df, _ = load_and_clean_data()

    tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "🔍 NL → Data", "📚 RAG Context"])

    # ── Tab 1: Chat ────────────────────────────────────────────────────────────
    with tab1:
        st.markdown(
            "Ask anything about sales data. The agent **automatically selects the right tool** "
            "and grounds its answer in real data."
        )

        # Init
        if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
        if "msg_history"   not in st.session_state: st.session_state.msg_history   = []

        # Render history
        for msg in st.session_state.msg_history:
            avatar = "🛒" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        # Input
        preset = st.session_state.pop("preset_query", None)
        user_input = st.chat_input("Ask about sales, forecasts, anomalies, trends...")
        query = preset or user_input

        if query:
            st.session_state.msg_history.append({"role": "user", "content": query})
            with st.chat_message("user", avatar="👤"):
                st.markdown(query)

            with st.chat_message("assistant", avatar="🛒"):
                with st.spinner("Selecting tool and generating response..."):
                    response = agent_respond(
                        df, query,
                        chat_history=st.session_state.chat_history,
                        api_key=api_key or st.session_state.get("groq_api_key", ""),
                    )
                st.markdown(response)

            st.session_state.msg_history.append({"role": "assistant", "content": response})
            st.session_state.chat_history += [
                {"role": "user",      "content": query},
                {"role": "assistant", "content": response},
            ]
            # Keep last 10 turns to avoid context overflow
            if len(st.session_state.chat_history) > 20:
                st.session_state.chat_history = st.session_state.chat_history[-20:]

        col_l, col_r = st.columns([1, 5])
        if col_l.button("🗑️ Clear"):
            st.session_state.chat_history = []
            st.session_state.msg_history  = []
            st.rerun()

    # ── Tab 2: NL → Data ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        Converts plain-English retail analytics questions directly to Pandas operations.
        **No API key required.** Results are returned instantly.
        """)

        nl_q = st.text_input(
            "Natural language query",
            value="Average sales for Store 5 in 2012",
            placeholder="Best performing department",
        )

        if st.button("Run Query", type="primary") or nl_q:
            with st.spinner("Parsing..."):
                result = nl_to_pandas(df, nl_q)

            st.markdown("**📊 Result:**")
            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True, hide_index=True)
            elif isinstance(result, pd.Series):
                st.dataframe(result.reset_index(), use_container_width=True)
            else:
                st.markdown(f"""
                <div style="background:#1e293b; border:1px solid #3b82f6; border-radius:8px;
                            padding:16px; color:#e2e8f0; font-size:1.1rem;">
                {result}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Supported Query Patterns:**")
        patterns = [
            ("Average sales for Store N in YYYY",      "Filter store + year → mean weekly sales"),
            ("Best performing department",              "Group by Dept → idxmax by total revenue"),
            ("Monthly sales trend [for YYYY]",         "Group by Month → avg weekly sales"),
            ("Top N stores",                           "nlargest by total revenue"),
            ("Sales in YYYY",                          "Filter year → sum"),
            ("Holiday vs non-holiday comparison",      "Mean + uplift % calculation"),
            ("Compare Store N and Store M",            "Side-by-side store summary"),
            ("Show Store N summary",                   "Revenue, avg, peak, top dept"),
        ]
        st.dataframe(
            pd.DataFrame(patterns, columns=["Query Pattern", "Pandas Operation"]),
            use_container_width=True, hide_index=True,
        )

    # ── Tab 3: RAG Context ────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        This is the **Retrieval-Augmented Generation (RAG)** context automatically
        injected into every LLM call. It grounds the model in real dataset statistics,
        preventing hallucination of sales figures.
        """)
        ctx = build_rag_context(df)
        st.code(ctx, language="text")

        st.markdown("---")
        st.markdown("""
        **RAG Pipeline:**
        1. **Retrieval:** Tool is called → returns computed data (e.g. store summary).
        2. **Augmentation:** Tool output + dataset context are prepended to the LLM prompt.
        3. **Generation:** LLaMA 3 (70B) synthesises a business narrative grounded in the data.

        This prevents the LLM from making up sales numbers and ensures responses
        are always anchored to actual computed values from the Walmart dataset.
        """)

        st.markdown("**Current session memory:**")
        n_turns = len(st.session_state.get("chat_history", [])) // 2
        st.metric("Conversation turns in context", n_turns)
        st.caption("Context is limited to the last 10 turns to stay within LLaMA 3's 8K context window.")
