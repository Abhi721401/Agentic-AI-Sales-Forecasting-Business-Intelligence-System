"""
modules/agent.py
────────────────
Agentic AI system + conversational chatbot powered by Groq / LLaMA 3.

Architecture:
  - Tool registry: 6 tools the agent can invoke
  - Router: LLM decides which tool(s) to call based on the query
  - RAG context: dataset summaries + cached results passed to every call
  - Chatbot: multi-turn conversation with persistent message history

The agent follows a ReAct-style loop:
  1. Reason about what the user wants.
  2. Act: select and call the appropriate tool.
  3. Observe: receive tool output.
  4. Respond: synthesise a natural language answer.
"""

import os
import json
import re
import pandas as pd
import numpy as np
import streamlit as st

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

MODEL = "llama-3.1-8b-instant"

# ── RAG context builder ────────────────────────────────────────────────────────

def build_rag_context(df: pd.DataFrame) -> str:
    """
    Construct a compact dataset summary to inject into every LLM call.
    Acts as the retrieval layer of the RAG pipeline.
    """
    stores    = sorted(df["Store"].unique().tolist())
    depts     = sorted(df["Dept"].unique().tolist())
    date_min  = df["Date"].min().strftime("%Y-%m-%d")
    date_max  = df["Date"].max().strftime("%Y-%m-%d")
    total_rev = df["Weekly_Sales"].sum()
    avg_sales = df["Weekly_Sales"].mean()
    top_store = df.groupby("Store")["Weekly_Sales"].sum().idxmax()
    top_dept  = df.groupby("Dept")["Weekly_Sales"].sum().idxmax()

    context = f"""
DATASET CONTEXT (Walmart Sales Forecasting):
- 45 stores, {len(depts)} departments, weekly data from {date_min} to {date_max}
- Stores: {stores[:10]}... (showing first 10)
- Total revenue in dataset: ${total_rev:,.0f}
- Average weekly sales per (store, dept): ${avg_sales:,.0f}
- Top performing store: Store {top_store}
- Top performing department: Dept {top_dept}
- Store types: A (large), B (medium), C (small)
- Holiday weeks have a weight multiplier in the Kaggle WMAE metric
- MarkDown data only available post Nov 2011
"""
    return context.strip()


# ── Tool definitions ───────────────────────────────────────────────────────────

def tool_query_sales(df: pd.DataFrame, query: str) -> str:
    """Tool 1: Answer natural-language sales queries with Pandas."""
    result = nl_to_pandas(df, query)
    return str(result)


def tool_store_summary(df: pd.DataFrame, store: int) -> str:
    """Tool 2: Summarise a specific store's performance."""
    s = df[df["Store"] == store]
    if s.empty:
        return f"No data found for Store {store}."
    total  = s["Weekly_Sales"].sum()
    avg    = s["Weekly_Sales"].mean()
    peak   = s.loc[s["Weekly_Sales"].idxmax()]
    top_d  = s.groupby("Dept")["Weekly_Sales"].sum().idxmax()
    return (
        f"Store {store} Summary:\n"
        f"  Total Revenue: ${total:,.0f}\n"
        f"  Avg Weekly Sales: ${avg:,.0f}\n"
        f"  Peak Week: {peak['Date'].date()} (${peak['Weekly_Sales']:,.0f})\n"
        f"  Top Department: {top_d}"
    )


def tool_compare_stores(df: pd.DataFrame, s1: int, s2: int) -> str:
    """Tool 3: Compare two stores side-by-side."""
    results = []
    for s in [s1, s2]:
        sub = df[df["Store"] == s]
        results.append({
            "Store":        s,
            "Total_Revenue": sub["Weekly_Sales"].sum(),
            "Avg_Weekly":   sub["Weekly_Sales"].mean(),
            "Volatility":   sub["Weekly_Sales"].std(),
            "Num_Depts":    sub["Dept"].nunique(),
        })
    r1, r2 = results
    winner = r1["Store"] if r1["Total_Revenue"] > r2["Total_Revenue"] else r2["Store"]
    return (
        f"Store {r1['Store']}: Total ${r1['Total_Revenue']:,.0f} | "
        f"Avg ${r1['Avg_Weekly']:,.0f} | {r1['Num_Depts']} depts\n"
        f"Store {r2['Store']}: Total ${r2['Total_Revenue']:,.0f} | "
        f"Avg ${r2['Avg_Weekly']:,.0f} | {r2['Num_Depts']} depts\n"
        f"Winner: Store {winner} by total revenue."
    )


def tool_trend_analysis(df: pd.DataFrame, year: int = None) -> str:
    """Tool 4: Month-over-month trend analysis."""
    sub = df.copy()
    if year:
        sub = sub[sub["Year"] == year]
    monthly = sub.groupby("Month")["Weekly_Sales"].mean()
    peak_m  = monthly.idxmax()
    months  = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    trend_str = " | ".join([f"{months.get(m,'?')}: ${v:,.0f}" for m, v in monthly.items()])
    return f"Monthly avg sales ({year or 'all years'}): {trend_str}\nPeak month: {months.get(peak_m,'?')}"


def tool_anomaly_report(df: pd.DataFrame) -> str:
    """Tool 5: Flag top anomalous periods."""
    if "IQR_Outlier" not in df.columns:
        return "Anomaly detection not yet run. Please visit the Anomaly Detection page first."
    outliers = df[df["IQR_Outlier"] == 1].nlargest(10, "Weekly_Sales")
    lines = []
    for _, row in outliers.iterrows():
        lines.append(f"  Store {row['Store']} Dept {row['Dept']} on {row['Date'].date()}: "
                     f"${row['Weekly_Sales']:,.0f} (Holiday={bool(row['IsHoliday'])})")
    return "Top 10 sales anomalies:\n" + "\n".join(lines)


def tool_forecast_summary(store: int, horizon: str = "12 weeks") -> str:
    """Tool 6: Placeholder forecast summary (full forecast runs in UI)."""
    return (
        f"Forecast for Store {store} over {horizon}: "
        f"Please navigate to the Forecasting page for interactive forecast charts and exact values. "
        f"The system uses ARIMA and SARIMA models with AIC-optimised orders."
    )


# ── Natural Language → Pandas ──────────────────────────────────────────────────

def nl_to_pandas(df: pd.DataFrame, query: str):
    """
    Rule-based NL → Pandas query converter.
    Handles the most common retail analytics questions.
    """
    q = query.lower()

    # Average sales for a store in a year
    m = re.search(r"average sales.+store\s*(\d+).+(\d{4})", q)
    if m:
        store, year = int(m.group(1)), int(m.group(2))
        val = df[(df["Store"] == store) & (df["Year"] == year)]["Weekly_Sales"].mean()
        return f"Store {store} average weekly sales in {year}: ${val:,.2f}"

    # Best performing department
    if "best performing department" in q or "top department" in q:
        top = df.groupby("Dept")["Weekly_Sales"].sum().idxmax()
        total = df.groupby("Dept")["Weekly_Sales"].sum().max()
        return f"Best performing department: Dept {top} with total sales ${total:,.0f}"

    # Monthly sales trend
    if "monthly sales" in q or "monthly trend" in q:
        monthly = df.groupby("Month")["Weekly_Sales"].mean().reset_index()
        monthly.columns = ["Month", "Avg_Weekly_Sales"]
        return monthly.to_string(index=False)

    # Top N stores
    m = re.search(r"top\s*(\d+)\s*stores?", q)
    if m:
        n = int(m.group(1))
        top = df.groupby("Store")["Weekly_Sales"].sum().nlargest(n).reset_index()
        return top.to_string(index=False)

    # Sales in year
    m = re.search(r"sales.+(\d{4})", q)
    if m:
        year = int(m.group(1))
        total = df[df["Year"] == year]["Weekly_Sales"].sum()
        return f"Total sales in {year}: ${total:,.0f}"

    # Holiday vs non-holiday
    if "holiday" in q and ("compare" in q or "vs" in q or "versus" in q):
        h  = df[df["IsHoliday"] == 1]["Weekly_Sales"].mean()
        nh = df[df["IsHoliday"] == 0]["Weekly_Sales"].mean()
        return f"Holiday avg: ${h:,.0f} | Non-holiday avg: ${nh:,.0f} | Uplift: {(h/nh-1)*100:.1f}%"

    # Store comparison
    m = re.search(r"store\s*(\d+).+store\s*(\d+)", q)
    if m:
        return tool_compare_stores(df, int(m.group(1)), int(m.group(2)))

    return "I couldn't parse that query. Try: 'Average sales for Store 5 in 2012' or 'Best performing department'."


# ── Agent router ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are RetailGPT, an expert AI analyst embedded in a Walmart sales analytics platform.
You have access to historical sales data for 45 Walmart stores from 2010-2012.
Your role is to answer questions about sales trends, forecasts, anomalies, and business insights.
Be concise, data-driven, and use specific numbers when available.
Always phrase insights as actionable business intelligence.
When you don't have exact numbers, explain what analysis would reveal the answer.
"""


def call_groq(messages: list, context: str = "", api_key: str = "") -> str:
    """Call Groq API with message history and RAG context."""
    if not GROQ_AVAILABLE or not api_key:
        return "⚠️ Groq API not configured. Add your GROQ_API_KEY to the sidebar settings."

    client = Groq(api_key=api_key)

    system_with_context = SYSTEM_PROMPT
    if context:
        system_with_context += f"\n\n{context}"

    full_messages = [{"role": "system", "content": system_with_context}] + messages

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"


def agent_respond(df: pd.DataFrame, user_query: str,
                  chat_history: list, api_key: str = "") -> str:
    """
    Agentic response: first try rule-based tools, then augment with LLM.
    Tool results are injected as context into the LLM call.
    """
    q = user_query.lower()
    tool_context = ""

    # Route to tools
    store_match = re.search(r"store\s*(\d+)", q)
    store = int(store_match.group(1)) if store_match else None

    if "compar" in q and store:
        m2 = re.findall(r"store\s*(\d+)", q)
        if len(m2) >= 2:
            tool_context = tool_compare_stores(df, int(m2[0]), int(m2[1]))
    elif "anomal" in q or "unusual" in q or "outlier" in q:
        tool_context = tool_anomaly_report(df)
    elif "forecast" in q and store:
        tool_context = tool_forecast_summary(store)
    elif store and ("summary" in q or "perform" in q or "revenue" in q):
        tool_context = tool_store_summary(df, store)
    elif "trend" in q or "monthly" in q:
        year_m = re.search(r"(\d{4})", q)
        year = int(year_m.group(1)) if year_m else None
        tool_context = tool_trend_analysis(df, year)
    else:
        tool_context = tool_query_sales(df, user_query)

    rag_context = build_rag_context(df)
    full_context = rag_context + "\n\nTOOL OUTPUT:\n" + tool_context

    augmented_history = chat_history + [{"role": "user", "content": user_query}]
    return call_groq(augmented_history, context=full_context, api_key=api_key)
