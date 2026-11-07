#!/usr/bin/env python
# HW10: Agents, Tools & Multimodal AI
# CIS 5300: Natural Language Processing | Module 13
# Build a ReAct agent using Anthropic's tool_use API

import anthropic
import json
import math
import os
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


# ============================================================
# Section 0: Setup
# ============================================================
# Install dependencies (guarded so the autograder can import this file
# without triggering a pip install; the notebook still installs on Run All).
if __name__ == '__main__':
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anthropic", "numpy"])


# ============================================================
# DO NOT MODIFY - Quick API test (guarded so importing hw10.py
# does NOT call the API)
# ============================================================
if __name__ == '__main__':
    assert os.environ.get("ANTHROPIC_API_KEY"), (
        "ANTHROPIC_API_KEY is not set. Configure it in your environment.")
    client = anthropic.Anthropic()
    test_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'API connection successful' and nothing else."}])
    print(test_response.content[0].text)
    print("\nSetup complete! You are ready to proceed.")


# ============================================================
# Section 1: Tool Definitions (15 points)
# ============================================================

# --- 1.1 Calculator Tool (5 points) ---

calculator_tool_schema = {
    "name": "calculator",
    "description": (
        "Evaluate a mathematical expression and return the numeric result. "
        "Supports arithmetic operations (+, -, *, /, **), parentheses, and "
        "math module functions (math.sqrt, math.log, math.sin, math.cos, "
        "math.pi, etc.). Use this tool whenever the question requires a "
        "precise numerical calculation rather than estimating or recalling "
        "a fact from memory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "A Python math expression to evaluate, e.g. "
                    "'2 + 3 * 4', 'math.sqrt(144)', '10 ** 2'. "
                    "Supports the math module (math.sqrt, math.pi, etc.)."
                )
            }
        },
        "required": ["expression"]
    }
}


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result as a string.

    Returns an 'Error: ...' string if the expression is invalid.

    Examples:
        >>> calculator("2 + 3")
        '5'
        >>> calculator("math.sqrt(144)")
        '12'
    """
    try:
        allowed = {"__builtins__": {}, "math": math}
        result = eval(expression, allowed)
        # Convert to a clean string representation
        if isinstance(result, float):
            if result == int(result):
                return str(int(result))
            return str(result)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# --- 1.2 Wikipedia Search Tool (5 points) ---

WIKI_DATA = {
    "france": "France is a country in Western Europe. Its capital is Paris. France has a population of approximately 67.75 million people (2024). The official language is French. France is a founding member of the European Union.",
    "paris": "Paris is the capital and largest city of France. It has a population of approximately 2.16 million in the city proper and over 12 million in the metropolitan area. Paris is known for the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
    "eiffel tower": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was designed by Gustave Eiffel and built between 1887 and 1889. The tower is 330 meters (1,083 feet) tall and weighs approximately 7,300 tonnes. It was the tallest man-made structure in the world until 1930.",
    "japan": "Japan is an island country in East Asia. Its capital is Tokyo. Japan has a population of approximately 125 million people. The official language is Japanese. Japan is known for its technology, culture, and cuisine.",
    "tokyo": "Tokyo is the capital and most populous city of Japan. It has a population of approximately 13.96 million in the city proper and over 37 million in the Greater Tokyo Area, making it the most populous metropolitan area in the world. Tokyo hosted the 2020 Summer Olympics.",
    "mount everest": "Mount Everest is the highest mountain above sea level on Earth. It is located in the Mahalangur Himal sub-range of the Himalayas, on the border between Nepal and Tibet (China). Its peak is 8,849 meters (29,032 feet) above sea level.",
    "python programming": "Python is a high-level, general-purpose programming language. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with significant indentation. It is widely used in web development, data science, AI, and scientific computing.",
    "albert einstein": "Albert Einstein (1879-1955) was a German-born theoretical physicist. He developed the theory of relativity, including E=mc^2. Einstein was born in Ulm, Germany, and later became a Swiss and American citizen. He won the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.",
    "speed of light": "The speed of light in vacuum is exactly 299,792,458 meters per second (approximately 300,000 km/s or 186,000 miles per second). It is denoted by the symbol c and is a fundamental physical constant. According to special relativity, no object with mass can travel at the speed of light.",
    "united states": "The United States of America is a country in North America. Its capital is Washington, D.C. The United States has a population of approximately 331 million people. It is the third-largest country by area and the third-most populous country in the world.",
    "washington dc": "Washington, D.C. is the capital of the United States. It has a population of approximately 689,545 (2020 census). The city is home to the White House, the Capitol Building, and the Supreme Court. D.C. stands for District of Columbia.",
    "amazon river": "The Amazon River is the largest river by discharge volume in the world. It flows through South America, primarily through Brazil, Peru, and Colombia. The Amazon is approximately 6,400 km (4,000 miles) long, making it one of the longest rivers in the world.",
    "photosynthesis": "Photosynthesis is the biological process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. The overall equation is: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. Photosynthesis occurs primarily in chloroplasts within plant cells.",
    "marie curie": "Marie Curie (1867-1934) was a Polish-born French physicist and chemist. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences (Physics in 1903 and Chemistry in 1911). She discovered the elements polonium and radium.",
    "pacific ocean": "The Pacific Ocean is the largest and deepest ocean on Earth. It covers an area of approximately 165.25 million square kilometers. The Mariana Trench, located in the western Pacific, is the deepest point on Earth at approximately 10,994 meters (36,070 feet).",
    "brazil": "Brazil is the largest country in South America with a population of approximately 214 million people. Its capital is Brasilia. The largest city is Sao Paulo. The official language is Portuguese. Brazil is known for the Amazon rainforest, Carnival, and football.",
    "sun": "The Sun is the star at the center of the Solar System. It has a diameter of approximately 1.39 million kilometers. The surface temperature of the Sun is about 5,778 Kelvin (5,505 degrees Celsius). Light from the Sun takes approximately 8 minutes and 20 seconds to reach Earth.",
    "earth": "Earth is the third planet from the Sun and the only known planet to support life. It has a diameter of approximately 12,742 km. Earth's distance from the Sun is approximately 149.6 million km (1 AU). The planet has one natural satellite, the Moon.",
}

wikipedia_search_tool_schema = {
    "name": "wikipedia_search",
    "description": (
        "Search a simulated Wikipedia database for encyclopedic information "
        "about people, places, concepts, and facts. Returns a summary of "
        "the best-matching article. Use this tool when the question asks "
        "about a fact, entity, or piece of knowledge that would appear in "
        "an encyclopedia (capitals, heights, populations, biographies, "
        "scientific constants, etc.)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query - a topic name, person name, or "
                    "question keywords to search for in the Wikipedia "
                    "database."
                )
            }
        },
        "required": ["query"]
    }
}


def wikipedia_search(query: str) -> str:
    """Search the simulated Wikipedia database (WIKI_DATA) and return the matched entry's TEXT.

    Find the best-matching key (exact case-insensitive match first, otherwise the key with
    the most shared words), then return WIKI_DATA[best_key] — the summary string stored
    under that key, NOT the key itself and NOT the original query. If no key matches at
    all, return a 'No results found' message instead.

    Examples:
        >>> wikipedia_search("France")
        'France is a country in Western Europe. Its capital is Paris. ...'
        >>> wikipedia_search("xyzzy nonsense query")
        'No Wikipedia results found for: xyzzy nonsense query'
    """
    if not query or not query.strip():
        return f"No Wikipedia results found for: {query}"

    query_lower = query.lower().strip()
    query_words = set(query_lower.split())

    # Try exact case-insensitive match first
    for key in WIKI_DATA:
        if key.lower() == query_lower:
            return WIKI_DATA[key]

    # Otherwise, find the key with the most shared words
    best_key = None
    best_overlap = 0

    for key in WIKI_DATA:
        key_words = set(key.lower().split())
        overlap = len(query_words & key_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key

    if best_key is not None and best_overlap > 0:
        return WIKI_DATA[best_key]

    return f"No Wikipedia results found for: {query}"


# --- 1.3 Weather Lookup Tool (5 points) ---

WEATHER_DATA = {
    "paris": {"temp_c": 18, "condition": "Partly cloudy", "humidity": 65},
    "tokyo": {"temp_c": 22, "condition": "Sunny", "humidity": 55},
    "new york": {"temp_c": 15, "condition": "Overcast", "humidity": 70},
    "london": {"temp_c": 12, "condition": "Rainy", "humidity": 85},
    "sydney": {"temp_c": 25, "condition": "Sunny", "humidity": 45},
    "berlin": {"temp_c": 14, "condition": "Cloudy", "humidity": 72},
    "washington": {"temp_c": 20, "condition": "Clear", "humidity": 58},
    "sao paulo": {"temp_c": 28, "condition": "Thunderstorms", "humidity": 80},
    "mumbai": {"temp_c": 32, "condition": "Humid and hazy", "humidity": 90},
    "cairo": {"temp_c": 35, "condition": "Sunny and dry", "humidity": 20},
}

weather_lookup_tool_schema = {
    "name": "weather_lookup",
    "description": (
        "Look up current weather conditions for a given city, including "
        "temperature, weather condition, and humidity. Use this tool when "
        "the question asks about weather, temperature, or conditions in a "
        "specific location. Supports optional unit conversion between "
        "Celsius and Fahrenheit."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "The city name to look up weather for, e.g. 'Paris', "
                    "'Tokyo', 'London'."
                )
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": (
                    "Temperature unit: 'celsius' (default) or "
                    "'fahrenheit'. Optional."
                )
            }
        },
        "required": ["location"]
    }
}


def weather_lookup(location: str, units: str = "celsius") -> str:
    """Look up current weather for a location from WEATHER_DATA.

    Convert to Fahrenheit if units=='fahrenheit' (F = C * 9/5 + 32).
    Return a 'not available' message for unknown locations.

    The return value is a single formatted sentence — nothing more elaborate than that.
    Match this exact shape (the sanity checks below rely on it):
        "Weather in {City}: {temp}°{C or F}, {condition}, Humidity: {humidity}%"
    e.g. weather_lookup("Paris") -> "Weather in Paris: 18°C, Partly cloudy, Humidity: 65%"
    """
    location_lower = location.lower().strip()

    # Find matching city (case-insensitive, partial match)
    matched_key = None
    for key in WEATHER_DATA:
        if location_lower in key or key in location_lower:
            matched_key = key
            break

    # Also try exact match
    if matched_key is None:
        for key in WEATHER_DATA:
            if key == location_lower:
                matched_key = key
                break

    if matched_key is None:
        return f"Weather not available for: {location}"

    data = WEATHER_DATA[matched_key]
    # Capitalize city name for display
    city_name = matched_key.title()

    if units == "fahrenheit":
        temp = data["temp_c"] * 9 / 5 + 32
        # Preserve decimals when conversion yields a non-integer
        if temp == int(temp):
            temp = int(temp)
        unit_label = "F"
    else:
        temp = data["temp_c"]
        unit_label = "C"

    return (f"Weather in {city_name}: {temp}°{unit_label}, "
            f"{data['condition']}, Humidity: {data['humidity']}%")


# --- Collect all tools ---

ALL_TOOL_SCHEMAS = [
    calculator_tool_schema,
    wikipedia_search_tool_schema,
    weather_lookup_tool_schema,
]

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "wikipedia_search": wikipedia_search,
    "weather_lookup": weather_lookup,
}

if __name__ == '__main__':
    print(f"Registered {len(ALL_TOOL_SCHEMAS)} tools:")
    for schema in ALL_TOOL_SCHEMAS:
        print(f"  - {schema.get('name','?')}: {str(schema.get('description',''))[:80]}...")


# ============================================================
# Section 2: ReAct Agent (33 points)
# ============================================================

# --- 2.1 Execute a Tool Call (5 points) ---

def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute a tool by name with the given input arguments; return a string.

    Steps:
      1. Look up `tool_name` in TOOL_FUNCTIONS. If it's not there, return an "Error: ..." string
         (don't raise) — list the available tools so the caller/model can see what's valid.
      2. Otherwise call the function with `**tool_input`, so a schema like {"expression": "2+3"}
         becomes calculator(expression="2+3").
      3. If the call raises ANY exception (e.g. a required argument is missing or malformed), catch
         it and return an "Error executing {tool_name}: ..." string instead of letting it propagate —
         one bad tool call should never crash the whole agent loop.
    """
    if tool_name not in TOOL_FUNCTIONS:
        available = ", ".join(TOOL_FUNCTIONS.keys())
        return f"Error: Tool '{tool_name}' not found. Available tools: {available}"

    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**tool_input)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


# --- 2.2 The ReAct Loop (25 points) ---

def run_agent(question: str, max_steps: int = 10, verbose: bool = True) -> Dict:
    """Run the ReAct agent on a question.

    Args:
        question: the user's question.
        max_steps: hard cap on Thought/Action/Observation cycles, so a model that keeps calling
            tools without converging can't loop forever. 10 is generous for this tool set — most
            questions here resolve in 1-2 tool calls.
        verbose: if True, print each step's thought/action/observation as the loop runs (useful for
            debugging); has no effect on the returned dict either way.

    Returns a dict with:
        'answer': str, 'steps': List[Dict] (each: thought, tool_name, tool_input, observation),
        'total_tool_calls': int, 'stopped_reason': 'answered' | 'max_steps' | 'error'
    """
    client = anthropic.Anthropic()
    system_prompt = (
        "You are a helpful assistant with access to tools. "
        "Use the provided tools to answer the user's question accurately. "
        "Think step by step about what information you need and which tool to use. "
        "When you have enough information, provide a clear final answer. "
        "If a tool returns an error, try a different approach."
    )
    messages = [{"role": "user", "content": question}]
    steps = []
    total_tool_calls = 0

    for step_num in range(max_steps):
        # 1. Call the API
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                temperature=0,
                system=system_prompt,
                tools=ALL_TOOL_SCHEMAS,
                messages=messages,
            )
        except Exception as e:
            return {
                "answer": f"API Error: {str(e)}",
                "steps": steps,
                "total_tool_calls": total_tool_calls,
                "stopped_reason": "error"
            }

        # 2. Collect text blocks and find tool_use block
        thought = ""
        tool_use_block = None

        for block in response.content:
            if hasattr(block, 'text'):
                thought += block.text
            elif hasattr(block, 'type') and block.type == "tool_use":
                tool_use_block = block

        # 3. If stop_reason == "end_turn" or no tool_use block: agent is done
        if response.stop_reason == "end_turn" or tool_use_block is None:
            if verbose:
                print(f"  [Final] {thought[:200]}")
            return {
                "answer": thought,
                "steps": steps,
                "total_tool_calls": total_tool_calls,
                "stopped_reason": "answered"
            }

        # 4. Execute the tool
        tool_name = tool_use_block.name
        tool_input = tool_use_block.input
        observation = execute_tool(tool_name, tool_input)

        if verbose:
            print(f"  Step {step_num + 1}:")
            print(f"    Thought: {thought[:200]}")
            print(f"    Action: {tool_name}({tool_input})")
            print(f"    Observation: {observation[:200]}")

        # Record the step
        steps.append({
            "thought": thought,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "observation": observation,
        })
        total_tool_calls += 1

        # Append assistant response and tool result to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": observation,
            }]
        })

    # Loop exhausted max_steps
    if verbose:
        print("  [Max steps reached]")
    return {
        "answer": thought if thought else "Max steps reached without an answer.",
        "steps": steps,
        "total_tool_calls": total_tool_calls,
        "stopped_reason": "max_steps"
    }


# --- 2.3 Error Recovery ---

def run_agent_robust(question: str, max_steps: int = 10, max_retries: int = 2,
                     verbose: bool = True) -> Dict:
    """Run the ReAct agent with explicit retry logic for tool failures.

    When a tool returns an "Error: ..." string, retry up to max_retries times
    before passing the observation to the model. This reduces reliance on the
    model noticing and recovering from errors on its own.
    """
    client = anthropic.Anthropic()
    system_prompt = (
        "You are a helpful assistant with access to tools. "
        "Use the provided tools to answer the user's question accurately. "
        "Think step by step about what information you need and which tool to use. "
        "When you have enough information, provide a clear final answer. "
        "If a tool returns an error, try a different approach."
    )
    messages = [{"role": "user", "content": question}]
    steps = []
    total_tool_calls = 0

    for step_num in range(max_steps):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                temperature=0,
                system=system_prompt,
                tools=ALL_TOOL_SCHEMAS,
                messages=messages,
            )
        except Exception as e:
            return {
                "answer": f"API Error: {str(e)}",
                "steps": steps,
                "total_tool_calls": total_tool_calls,
                "stopped_reason": "error"
            }

        thought = ""
        tool_use_block = None

        for block in response.content:
            if hasattr(block, 'text'):
                thought += block.text
            elif hasattr(block, 'type') and block.type == "tool_use":
                tool_use_block = block

        if response.stop_reason == "end_turn" or tool_use_block is None:
            if verbose:
                print(f"  [Final] {thought[:200]}")
            return {
                "answer": thought,
                "steps": steps,
                "total_tool_calls": total_tool_calls,
                "stopped_reason": "answered"
            }

        tool_name = tool_use_block.name
        tool_input = tool_use_block.input

        # Execute tool with retry logic
        observation = execute_tool(tool_name, tool_input)
        retry_count = 0

        while observation.startswith("Error") and retry_count < max_retries:
            retry_count += 1
            if verbose:
                print(f"    [Retry {retry_count}/{max_retries}] for {tool_name}")
            observation = execute_tool(tool_name, tool_input)

        if verbose:
            print(f"  Step {step_num + 1}:")
            print(f"    Thought: {thought[:200]}")
            print(f"    Action: {tool_name}({tool_input})")
            print(f"    Observation: {observation[:200]}")

        steps.append({
            "thought": thought,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "observation": observation,
        })
        total_tool_calls += 1

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": observation,
            }]
        })

    if verbose:
        print("  [Max steps reached]")
    return {
        "answer": thought if thought else "Max steps reached without an answer.",
        "steps": steps,
        "total_tool_calls": total_tool_calls,
        "stopped_reason": "max_steps"
    }


# ============================================================
# Section 3: Agent Evaluation (20 points)
# ============================================================

EVAL_QUESTIONS = [
    {"question": "What is 247 * 18?", "expected_answer": "4446", "expected_tool": "calculator", "difficulty": "easy"},
    {"question": "What is the square root of 1764?", "expected_answer": "42", "expected_tool": "calculator", "difficulty": "easy"},
    {"question": "What is the capital of Japan?", "expected_answer": "tokyo", "expected_tool": "wikipedia_search", "difficulty": "easy"},
    {"question": "How tall is the Eiffel Tower in meters?", "expected_answer": "330", "expected_tool": "wikipedia_search", "difficulty": "easy"},
    {"question": "What is the weather in London?", "expected_answer": "rainy", "expected_tool": "weather_lookup", "difficulty": "easy"},
    {"question": "What is the temperature in Cairo in Fahrenheit?", "expected_answer": "95", "expected_tool": "weather_lookup", "difficulty": "medium"},
    {"question": "Who won the Nobel Prize for discovering radium?", "expected_answer": "curie", "expected_tool": "wikipedia_search", "difficulty": "medium"},
    {"question": "What is the speed of light in kilometers per second?", "expected_answer": "300000", "expected_tool": "wikipedia_search", "difficulty": "medium"},
    {"question": "If the Eiffel Tower is 330 meters tall, how many feet is that? (1 meter = 3.281 feet)", "expected_answer": "1082", "expected_tool": "calculator", "difficulty": "medium"},
    {"question": "What is the population of the largest city in Brazil?", "expected_answer": "sao paulo", "expected_tool": "wikipedia_search", "difficulty": "medium"},
]


def evaluate_agent(questions: List[Dict], verbose: bool = False) -> List[Dict]:
    """Evaluate the agent on questions with known answers.

    Each `questions[i]` is a dict with keys "question", "expected_answer", "expected_tool",
    "difficulty" (see EVAL_QUESTIONS above).

    For each question:
      - Run run_agent(q["question"], verbose=verbose) to get agent_result.
      - `correct`: True if q["expected_answer"].lower() is a SUBSTRING of
        agent_result["answer"].lower() — the same "in" pattern used by the sanity-check cells above
        (e.g. "42" in "the square root of 1764 is 42."). You don't need exact string equality.
      - `tools_used`: pull tool_name out of each dict in agent_result["steps"] into a list — you need
        this to tell "wrong_tool" apart from "wrong_answer" below.
      - `failure_type` (only set when NOT correct): "max_steps" or "error" if
        agent_result["stopped_reason"] says so; else "no_tool_used" if tools_used is empty; else
        "wrong_tool" if q["expected_tool"] not in tools_used; else "wrong_answer".

    Returns a list of dicts: question, expected_answer, agent_answer, correct (bool),
    tool_calls (int), tools_used (List[str]), stopped_reason, failure_type (or None).
    """
    results = []

    for q in questions:
        agent_result = run_agent(q["question"], verbose=verbose)

        answer = agent_result.get("answer", "")
        expected = q["expected_answer"].lower()
        correct = expected in answer.lower()

        tools_used = [step["tool_name"] for step in agent_result.get("steps", [])]
        tool_calls = agent_result.get("total_tool_calls", 0)
        stopped_reason = agent_result.get("stopped_reason", "")

        # Determine failure type if not correct
        failure_type = None
        if not correct:
            if stopped_reason in ("max_steps", "error"):
                failure_type = stopped_reason
            elif len(tools_used) == 0:
                failure_type = "no_tool_used"
            elif q["expected_tool"] not in tools_used:
                failure_type = "wrong_tool"
            else:
                failure_type = "wrong_answer"

        results.append({
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "agent_answer": answer,
            "correct": correct,
            "tool_calls": tool_calls,
            "tools_used": tools_used,
            "stopped_reason": stopped_reason,
            "failure_type": failure_type,
        })

    return results


# ============================================================
# Section 4: Multi-Step Reasoning (15 points)
# ============================================================

MULTI_STEP_QUESTIONS = [
    {"question": "What is the weather in the capital of France?", "expected_answer": "18",
     "required_tools": ["wikipedia_search", "weather_lookup"],
     "reasoning": "Find that Paris is the capital, then look up Paris weather"},
    {"question": "What is the height of the Eiffel Tower in feet? Calculate it given that 1 meter = 3.281 feet.",
     "expected_answer": "1082", "required_tools": ["wikipedia_search", "calculator"],
     "reasoning": "Find the height in meters (330m), then convert"},
    {"question": "What is the temperature difference between Tokyo and London in Celsius?",
     "expected_answer": "10", "required_tools": ["weather_lookup", "weather_lookup"],
     "reasoning": "Look up both cities, then compute the difference"},
    {"question": "How many years did it take to build the Eiffel Tower? It was built between 1887 and 1889.",
     "expected_answer": "2", "required_tools": ["wikipedia_search", "calculator"],
     "reasoning": "Find the dates, then calculate the difference"},
    {"question": "Albert Einstein was born in Ulm, Germany. What is the weather in the country where he was born?",
     "expected_answer": "berlin", "required_tools": ["wikipedia_search", "weather_lookup"],
     "reasoning": "Find the birthplace country (Germany), then look up weather in a German city"},
]


def run_multi_step_evaluation(questions: List[Dict], verbose: bool = True) -> List[Dict]:
    """Run the agent on multi-step questions and record results with full traces."""
    results = []

    for q in questions:
        agent_result = run_agent(q["question"], verbose=verbose)

        answer = agent_result.get("answer", "")
        expected = q["expected_answer"].lower()
        correct = expected in answer.lower()

        tools_used = [step["tool_name"] for step in agent_result.get("steps", [])]

        results.append({
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "agent_answer": answer,
            "correct": correct,
            "tool_calls": agent_result.get("total_tool_calls", 0),
            "tools_used": tools_used,
            "stopped_reason": agent_result.get("stopped_reason", ""),
            "steps": agent_result.get("steps", []),
            "reasoning": q.get("reasoning", ""),
            "required_tools": q.get("required_tools", []),
        })

    return results


if __name__ == '__main__':
    # --- Sanity checks for calculator ---
    assert calculator("2 + 3") == "5", f"Expected '5', got '{calculator('2 + 3')}'"
    assert calculator("10 * 5") == "50"
    assert float(calculator("math.sqrt(144)")) == 12.0
    assert calculator("invalid_expression").startswith("Error")
    assert calculator_tool_schema["name"] == "calculator"
    assert "expression" in calculator_tool_schema["input_schema"]["properties"]
    assert "expression" in calculator_tool_schema["input_schema"]["required"]
    print("Calculator checks passed!")

    # --- Sanity checks for wikipedia_search ---
    assert "Paris" in wikipedia_search("France")
    assert "Nobel" in wikipedia_search("albert einstein")
    assert "no" in wikipedia_search("xyzzy nonsense query").lower()
    assert "no" in wikipedia_search("").lower()
    assert wikipedia_search_tool_schema["name"] == "wikipedia_search"
    assert "query" in wikipedia_search_tool_schema["input_schema"]["properties"]
    print("Wikipedia search checks passed!")

    # --- Sanity checks for weather_lookup ---
    assert "18" in weather_lookup("Paris")
    assert "64" in weather_lookup("Paris", "fahrenheit")
    assert "not available" in weather_lookup("Atlantis").lower()
    assert weather_lookup_tool_schema["name"] == "weather_lookup"
    assert "location" in weather_lookup_tool_schema["input_schema"]["required"]
    assert "units" not in weather_lookup_tool_schema["input_schema"].get("required", [])
    print("Weather lookup checks passed!")

    # --- Sanity checks for execute_tool ---
    assert execute_tool("calculator", {"expression": "2 + 3"}) == "5"
    assert "Paris" in execute_tool("wikipedia_search", {"query": "France"})
    assert "18" in execute_tool("weather_lookup", {"location": "Paris"})
    assert "error" in execute_tool("nonexistent_tool", {}).lower() or \
           "not found" in execute_tool("nonexistent_tool", {}).lower()
    print("execute_tool tests passed!")

    print("\nAll sanity checks passed!")
