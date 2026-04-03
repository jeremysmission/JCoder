"""
Download code quality, linting, and best-practices datasets.

Focus: syntax errors, common mistakes, code smells, linting rules,
style guides, and beginner-friendly explanations.

Datasets:
  1. Python official docs (PEP 8, tutorial, FAQ, errors)
  2. Common Python mistakes and fixes (curated from known sources)
  3. Pylint/flake8 rule explanations
  4. Code smell patterns with fixes

These get indexed as high-weight references that JCoder can cite
when reviewing code or suggesting corrections.

Usage:
    cd C:\\Users\\jerem\\JCoder
    .venv\\Scripts\\python scripts\\download_best_practices.py
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

import httpx

DATA_ROOT = Path(os.environ.get("JCODER_DATA", "data"))
INDEX_DIR = DATA_ROOT / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2000
MAX_CHARS = 4000

_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_CAMEL_RE1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _normalize(text: str) -> str:
    out = _NORMALIZE_RE.sub(" ", text)
    out = _CAMEL_RE1.sub(r"\1 \2", out)
    out = _CAMEL_RE2.sub(r"\1 \2", out)
    return out.lower()


class FTS5Builder:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
            "USING fts5(search_content, source_path, chunk_id)"
        )
        self._batch = []
        self.total_chunks = 0
        self.total_entries = 0

    def add(self, text: str, source_id: str):
        if not text or len(text.strip()) < 20:
            return
        self.total_entries += 1
        pos = 0
        cidx = 0
        while pos < len(text):
            end = min(pos + MAX_CHARS, len(text))
            if end < len(text):
                nl = text.rfind("\n", pos, end)
                if nl > pos:
                    end = nl + 1
            chunk = text[pos:end]
            if chunk.strip():
                cid = hashlib.sha256(f"{source_id}:{cidx}".encode()).hexdigest()
                self._batch.append((_normalize(chunk), source_id, cid))
                cidx += 1
            pos = end
        if len(self._batch) >= BATCH_SIZE:
            self._flush()

    def _flush(self):
        if self._batch:
            self.conn.executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id) "
                "VALUES (?, ?, ?)", self._batch)
            self.conn.commit()
            self.total_chunks += len(self._batch)
            self._batch = []

    def finish(self):
        self._flush()
        self.conn.close()
        size_mb = self.db_path.stat().st_size / 1e6
        return self.total_entries, self.total_chunks, size_mb


# -----------------------------------------------------------------------
# Python best practices knowledge base (built-in, no download needed)
# -----------------------------------------------------------------------

PYTHON_BEST_PRACTICES = [
    # --- Syntax errors and fixes ---
    {
        "title": "SyntaxError: Missing colon after if/for/while/def/class",
        "content": """Common beginner mistake: forgetting the colon at the end of compound statements.

WRONG:
    if x > 5
        print(x)

    for i in range(10)
        print(i)

    def my_function()
        pass

FIX: Add a colon (:) at the end of the line:
    if x > 5:
        print(x)

    for i in range(10):
        print(i)

    def my_function():
        pass

RULE: Every if, elif, else, for, while, def, class, with, and try statement MUST end with a colon."""
    },
    {
        "title": "IndentationError: Inconsistent indentation",
        "content": """Python uses indentation to define code blocks. Mixing tabs and spaces causes errors.

WRONG (mixed tabs and spaces):
    def foo():
        x = 1       # spaces
    	y = 2       # tab

FIX: Use 4 spaces consistently. Configure your editor to convert tabs to spaces.

    def foo():
        x = 1
        y = 2

BEST PRACTICE: Set your editor to:
  - Use spaces, not tabs
  - Tab width = 4 spaces
  - Show whitespace characters (helps catch mixing)

PEP 8 says: Use 4 spaces per indentation level."""
    },
    {
        "title": "NameError: Variable not defined / typo in variable name",
        "content": """NameError means Python cannot find a variable with that name.

Common causes:
1. Typo in variable name:
    WRONG: pritn("hello")     # typo: pritn vs print
    FIX:   print("hello")

2. Using variable before assignment:
    WRONG: print(total)       # total not defined yet
           total = 0
    FIX:   total = 0
           print(total)

3. Wrong scope (variable inside function not accessible outside):
    WRONG:
        def calc():
            result = 42
        print(result)         # NameError: result not defined
    FIX:
        def calc():
            return 42
        result = calc()
        print(result)

BEST PRACTICE: Use descriptive variable names to reduce typos. Use an IDE with autocomplete."""
    },
    {
        "title": "TypeError: Wrong type passed to function",
        "content": """TypeError means you passed the wrong type of data to a function or operator.

Common cases:
1. Concatenating string + number:
    WRONG: "Age: " + 25           # can't add str + int
    FIX:   "Age: " + str(25)      # convert int to str
    BETTER: f"Age: {25}"          # use f-string

2. Calling a non-callable:
    WRONG: x = 5
           x()                    # int is not callable
    FIX:   Check if you accidentally overwrote a function name

3. Wrong number of arguments:
    WRONG: def greet(name):
               return f"Hi {name}"
           greet()                # missing required argument
    FIX:   greet("Alice")

BEST PRACTICE: Use type hints to catch these early:
    def greet(name: str) -> str:
        return f"Hi {name}" """
    },
    {
        "title": "IndexError: List index out of range",
        "content": """IndexError means you tried to access a list element that does not exist.

WRONG:
    fruits = ["apple", "banana", "cherry"]
    print(fruits[3])    # IndexError! Valid indexes are 0, 1, 2

FIX: Remember Python lists are 0-indexed. A list of length 3 has indexes 0, 1, 2.
    print(fruits[2])    # "cherry" (last element)
    print(fruits[-1])   # "cherry" (negative index = from end)

SAFE PATTERN: Check length before accessing:
    if len(fruits) > 3:
        print(fruits[3])
    else:
        print("Not enough fruits")

BEST PRACTICE: Use try/except for risky index access:
    try:
        item = my_list[index]
    except IndexError:
        item = default_value"""
    },
    {
        "title": "KeyError: Dictionary key not found",
        "content": """KeyError means you tried to access a dictionary key that does not exist.

WRONG:
    user = {"name": "Alice", "age": 30}
    print(user["email"])    # KeyError: 'email'

FIX OPTION 1 - Use .get() with a default:
    print(user.get("email", "no email"))    # returns "no email"

FIX OPTION 2 - Check first:
    if "email" in user:
        print(user["email"])

FIX OPTION 3 - Use try/except:
    try:
        print(user["email"])
    except KeyError:
        print("no email on file")

BEST PRACTICE: Always use .get() for optional keys. Only use [] for keys you KNOW exist."""
    },
    # --- Best practices ---
    {
        "title": "Python naming conventions (PEP 8)",
        "content": """PEP 8 naming conventions -- follow these for readable, professional code:

VARIABLES and FUNCTIONS: snake_case (lowercase with underscores)
    user_name = "Alice"
    def calculate_total(items):
        pass

CLASSES: PascalCase (capitalize each word, no underscores)
    class ShoppingCart:
        pass
    class HTTPConnection:
        pass

CONSTANTS: UPPER_SNAKE_CASE
    MAX_RETRIES = 3
    DATABASE_URL = "localhost:5432"

PRIVATE: prefix with underscore
    _internal_cache = {}
    def _helper_function():
        pass

AVOID:
    - Single letter names (except i, j, k for loop counters)
    - Names that shadow builtins: list, dict, str, id, type, input
    - CamelCase for functions (that's Java style, not Python)
    - Names starting with double underscore (name mangling)"""
    },
    {
        "title": "Mutable default arguments trap",
        "content": """One of Python's most common gotchas: mutable default arguments are shared across calls.

WRONG (dangerous):
    def add_item(item, items=[]):    # BAD! list is created ONCE
        items.append(item)
        return items

    add_item("a")  # returns ["a"]
    add_item("b")  # returns ["a", "b"] -- surprise!

FIX: Use None as default, create new list inside:
    def add_item(item, items=None):
        if items is None:
            items = []
        items.append(item)
        return items

    add_item("a")  # returns ["a"]
    add_item("b")  # returns ["b"] -- correct!

RULE: Never use [], {}, or set() as default argument values.
Always use None and create the mutable object inside the function."""
    },
    {
        "title": "String formatting best practices",
        "content": """Python has 4 ways to format strings. Use f-strings (Python 3.6+).

OLD WAY 1 - % formatting (avoid):
    "Hello %s, age %d" % (name, age)

OLD WAY 2 - .format() (OK but verbose):
    "Hello {}, age {}".format(name, age)

BEST - f-strings (use this):
    f"Hello {name}, age {age}"

f-strings can do expressions:
    f"Total: {price * quantity:.2f}"
    f"Name: {name.upper()}"
    f"Items: {len(cart)}"

MULTILINE f-strings:
    message = (
        f"Dear {name},\\n"
        f"Your order #{order_id} "
        f"totals ${total:.2f}."
    )

RULE: Use f-strings for all new code. They are faster and more readable."""
    },
    {
        "title": "File handling best practices",
        "content": """Always use 'with' statement for file operations. It automatically closes the file.

WRONG (file might not be closed on error):
    f = open("data.txt")
    content = f.read()
    f.close()

RIGHT:
    with open("data.txt") as f:
        content = f.read()
    # file is automatically closed here, even if an error occurs

WRITING:
    with open("output.txt", "w") as f:
        f.write("Hello\\n")

READING LINES:
    with open("data.txt") as f:
        for line in f:          # memory-efficient line iteration
            process(line.strip())

ENCODING (important on Windows):
    with open("data.txt", encoding="utf-8") as f:
        content = f.read()

RULE: Never use open() without 'with'. Always specify encoding for text files."""
    },
    {
        "title": "List comprehension vs loops",
        "content": """List comprehensions are faster and more Pythonic than manual loops for building lists.

BEFORE (loop):
    squares = []
    for x in range(10):
        squares.append(x ** 2)

AFTER (comprehension):
    squares = [x ** 2 for x in range(10)]

WITH FILTER:
    evens = [x for x in range(20) if x % 2 == 0]

NESTED (flatten):
    flat = [x for row in matrix for x in row]

DICT COMPREHENSION:
    word_lengths = {w: len(w) for w in words}

SET COMPREHENSION:
    unique_lengths = {len(w) for w in words}

WHEN NOT TO USE:
    - Complex logic (more than one if/for) -- use a regular loop
    - Side effects (printing, writing files) -- use a regular loop
    - When readability suffers -- clarity beats cleverness

RULE: If the comprehension fits on one line and is easy to read, use it.
Otherwise, use a regular loop."""
    },
    {
        "title": "Exception handling best practices",
        "content": """Handle specific exceptions, never catch everything blindly.

WRONG (catches ALL errors, hides bugs):
    try:
        result = do_something()
    except:
        pass

WRONG (too broad):
    try:
        result = do_something()
    except Exception:
        pass

RIGHT (specific exception):
    try:
        result = int(user_input)
    except ValueError:
        print("Please enter a valid number")

MULTIPLE EXCEPTIONS:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("Invalid JSON")
    except TypeError:
        print("Input must be a string")

CLEANUP (finally):
    try:
        conn = connect_db()
        result = conn.query(sql)
    except DatabaseError as e:
        log_error(e)
        result = None
    finally:
        conn.close()  # always runs

RULES:
  1. Catch specific exceptions, not bare except
  2. Don't use try/except for flow control
  3. Log or handle the error -- never silently pass
  4. Keep try blocks small (only the risky line)"""
    },
    {
        "title": "Import organization (PEP 8)",
        "content": """Organize imports in this order, with blank lines between groups:

1. Standard library imports
2. Third-party imports
3. Local/project imports

EXAMPLE:
    import os
    import sys
    from pathlib import Path

    import httpx
    import yaml
    from pydantic import BaseModel

    from myproject.config import Settings
    from myproject.utils import helper

RULES:
  - One import per line (not: import os, sys)
  - Use absolute imports (not relative: from . import foo)
  - Avoid wildcard imports (from module import *)
  - Put imports at top of file, after module docstring
  - Sort alphabetically within each group

TOOLS: Use isort or ruff to auto-sort imports:
    pip install ruff
    ruff check --select I --fix ."""
    },
    {
        "title": "Common comparison mistakes",
        "content": """Python comparison gotchas that trip up beginners.

1. == vs is:
    WRONG: if x is 5:           # 'is' checks identity, not equality
    RIGHT: if x == 5:           # '==' checks value equality
    EXCEPTION: Use 'is' only for None, True, False:
        if x is None:
        if x is True:

2. Chained comparisons (Python supports this!):
    VERBOSE: if x > 0 and x < 10:
    BETTER:  if 0 < x < 10:

3. Truthy/falsy values:
    WRONG: if len(my_list) > 0:
    RIGHT: if my_list:          # empty list is falsy
    WRONG: if x == True:
    RIGHT: if x:
    WRONG: if x == None:
    RIGHT: if x is None:

4. Comparing floats:
    WRONG: if 0.1 + 0.2 == 0.3:    # False! floating point math
    RIGHT: if abs((0.1 + 0.2) - 0.3) < 1e-9:
    BETTER: import math
            if math.isclose(0.1 + 0.2, 0.3):"""
    },
    {
        "title": "Function design best practices",
        "content": """Write clean, maintainable functions following these rules.

1. SINGLE RESPONSIBILITY: Each function does ONE thing.
    WRONG: def process_and_save_and_email(data): ...
    RIGHT: def process(data): ...
           def save(data): ...
           def send_email(data): ...

2. SMALL FUNCTIONS: Aim for 20 lines or less. If longer, split it.

3. DESCRIPTIVE NAMES: Verb + noun pattern.
    WRONG: def data(x): ...
    RIGHT: def calculate_total(items): ...
           def validate_email(address): ...
           def fetch_user_profile(user_id): ...

4. TYPE HINTS: Document what goes in and comes out.
    def calculate_total(items: list[float], tax_rate: float = 0.1) -> float:
        subtotal = sum(items)
        return subtotal * (1 + tax_rate)

5. DOCSTRINGS: Explain what, not how.
    def fetch_user(user_id: int) -> dict:
        \"\"\"Fetch user profile from the database.

        Returns empty dict if user not found.
        \"\"\"

6. RETURN EARLY: Avoid deep nesting.
    WRONG:
        def process(x):
            if x is not None:
                if x > 0:
                    return x * 2
            return 0

    RIGHT:
        def process(x):
            if x is None:
                return 0
            if x <= 0:
                return 0
            return x * 2"""
    },
    {
        "title": "Git best practices for beginners",
        "content": """Essential Git workflow for new programmers.

DAILY WORKFLOW:
    git status                  # see what changed
    git add file1.py file2.py  # stage specific files
    git commit -m "Add login validation"  # commit with clear message
    git push                    # push to remote

COMMIT MESSAGE FORMAT:
    verb + what changed (present tense, imperative mood)
    GOOD: "Add user authentication"
    GOOD: "Fix null pointer in parser"
    GOOD: "Update requirements for Python 3.12"
    BAD:  "fixed stuff"
    BAD:  "WIP"
    BAD:  "asdfasdf"

BRANCHING:
    git checkout -b feature/login   # create feature branch
    # ... make changes, commit ...
    git push -u origin feature/login
    # create pull request on GitHub

UNDO MISTAKES:
    git checkout -- file.py    # discard uncommitted changes to file
    git reset HEAD file.py     # unstage a file
    git stash                  # temporarily shelve changes
    git stash pop              # bring them back

RULES:
  - Commit often, push daily
  - Never commit secrets (API keys, passwords, .env files)
  - Use .gitignore for: __pycache__, .env, node_modules, *.pyc
  - Write meaningful commit messages
  - One logical change per commit"""
    },
    {
        "title": "Virtual environments (venv) for Python",
        "content": """Always use a virtual environment for Python projects. It isolates dependencies.

CREATE:
    python -m venv .venv              # creates .venv directory

ACTIVATE:
    # Windows:
    .venv\\Scripts\\activate
    # Mac/Linux:
    source .venv/bin/activate

INSTALL PACKAGES:
    pip install requests flask        # install into venv only
    pip freeze > requirements.txt     # save dependencies

REPRODUCE:
    pip install -r requirements.txt   # install from file

DEACTIVATE:
    deactivate

WHY:
  - Project A needs requests 2.28, Project B needs 2.31
  - Without venv, they conflict. With venv, each project has its own
  - Keeps your system Python clean
  - Makes deployment reproducible

RULES:
  - One venv per project
  - Add .venv/ to .gitignore
  - Always activate before installing packages
  - Keep requirements.txt updated"""
    },
]

# -----------------------------------------------------------------------
# JavaScript / TypeScript best practices
# -----------------------------------------------------------------------

JS_BEST_PRACTICES = [
    {
        "title": "JavaScript: var vs let vs const",
        "content": """Use const by default, let when you need to reassign, never var.

WRONG (var has function scope, causes bugs):
    var x = 5;
    if (true) {
        var x = 10;   // overwrites outer x!
    }
    console.log(x);   // 10 (surprise!)

RIGHT (let has block scope):
    let x = 5;
    if (true) {
        let x = 10;   // separate variable
    }
    console.log(x);   // 5 (correct)

BEST (const for values that won't change):
    const API_URL = "https://api.example.com";
    const users = [];         // array ref is const, contents can change
    users.push("Alice");      // OK
    // users = [];            // Error: can't reassign

RULE: const > let > never var"""
    },
    {
        "title": "JavaScript: === vs ==",
        "content": """Always use === (strict equality). Never use == (loose equality).

== does type coercion (causes surprises):
    0 == ""        // true (both coerce to 0)
    0 == false     // true
    "" == false    // true
    null == undefined  // true
    "5" == 5       // true

=== checks type AND value (predictable):
    0 === ""       // false
    0 === false    // false
    "5" === 5      // false

RULE: Always use === and !==. Configure ESLint eqeqeq rule."""
    },
    {
        "title": "JavaScript: async/await best practices",
        "content": """Use async/await instead of .then() chains. Always handle errors.

WRONG (callback hell):
    fetch(url)
        .then(res => res.json())
        .then(data => {
            return fetch(url2);
        })
        .then(res => res.json())
        .then(data2 => { ... })
        .catch(err => console.log(err));

RIGHT (async/await):
    async function fetchData() {
        try {
            const res1 = await fetch(url);
            const data1 = await res1.json();

            const res2 = await fetch(url2);
            const data2 = await res2.json();

            return { data1, data2 };
        } catch (error) {
            console.error("Fetch failed:", error);
            throw error;
        }
    }

PARALLEL (when requests are independent):
    const [data1, data2] = await Promise.all([
        fetch(url1).then(r => r.json()),
        fetch(url2).then(r => r.json()),
    ]);

RULE: Always wrap await in try/catch. Use Promise.all for parallel work."""
    },
]

# -----------------------------------------------------------------------
# General programming best practices
# -----------------------------------------------------------------------

GENERAL_BEST_PRACTICES = [
    {
        "title": "DRY principle: Don't Repeat Yourself",
        "content": """If you write the same code twice, extract it into a function.

BEFORE (repeated code):
    # In handler A
    user = db.get_user(id)
    if user is None:
        return {"error": "User not found"}, 404
    if not user.is_active:
        return {"error": "User inactive"}, 403

    # In handler B (same code again!)
    user = db.get_user(id)
    if user is None:
        return {"error": "User not found"}, 404
    if not user.is_active:
        return {"error": "User inactive"}, 403

AFTER (extracted to function):
    def get_active_user(user_id):
        user = db.get_user(user_id)
        if user is None:
            raise NotFoundError("User not found")
        if not user.is_active:
            raise ForbiddenError("User inactive")
        return user

    # Handler A
    user = get_active_user(id)
    # Handler B
    user = get_active_user(id)

RULE: Three occurrences = extract to function. Two = watch for a third."""
    },
    {
        "title": "KISS principle: Keep It Simple",
        "content": """Write the simplest code that works. Don't over-engineer.

OVER-ENGINEERED:
    class AbstractDataProcessorFactory:
        def create_processor(self, strategy_config):
            strategy = StrategyRegistry.get(strategy_config.type)
            return ProcessorBuilder(strategy).with_config(strategy_config).build()

SIMPLE (does the same thing):
    def process_data(data, method="default"):
        if method == "fast":
            return fast_process(data)
        return default_process(data)

GUIDELINES:
  - Don't add features "just in case"
  - Don't create abstractions for one use case
  - Don't use design patterns where a function will do
  - If a junior developer can't understand it, simplify it

RULE: The best code is the code you don't write."""
    },
    {
        "title": "Code review checklist for beginners",
        "content": """Use this checklist when reviewing your own code before committing:

CORRECTNESS:
  [ ] Does it work for normal inputs?
  [ ] Does it handle edge cases (empty, None, zero, negative)?
  [ ] Are there off-by-one errors in loops/slices?

READABILITY:
  [ ] Are variable names descriptive?
  [ ] Is the logic easy to follow?
  [ ] Would I understand this in 6 months?

SAFETY:
  [ ] No hardcoded passwords or API keys?
  [ ] User input validated/sanitized?
  [ ] Files and connections properly closed?

STYLE:
  [ ] Consistent formatting (run your linter)?
  [ ] No commented-out code left behind?
  [ ] Imports organized?

TESTS:
  [ ] Did I test the happy path?
  [ ] Did I test error cases?
  [ ] Do existing tests still pass?"""
    },
]


def main():
    db_path = INDEX_DIR / "best_practices.fts5.db"

    if db_path.exists() and db_path.stat().st_size > 100_000:
        print(f"[OK] {db_path.name} already exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print("=" * 60)
    print("Building Best Practices FTS5 Index")
    print(f"Index: {db_path}")
    print("=" * 60)

    builder = FTS5Builder(db_path)

    all_entries = (
        [("python_bp", e) for e in PYTHON_BEST_PRACTICES]
        + [("js_bp", e) for e in JS_BEST_PRACTICES]
        + [("general_bp", e) for e in GENERAL_BEST_PRACTICES]
    )

    for prefix, entry in all_entries:
        title = entry["title"]
        content = entry["content"]
        text = f"# {title}\n\n{content}"
        builder.add(text, f"{prefix}_{builder.total_entries:04d}")

    entries, chunks, size_mb = builder.finish()
    print(f"\n[OK] {db_path.name}: {entries} entries, {chunks} chunks ({size_mb:.1f} MB)")
    print(f"  Python practices: {len(PYTHON_BEST_PRACTICES)}")
    print(f"  JavaScript practices: {len(JS_BEST_PRACTICES)}")
    print(f"  General practices: {len(GENERAL_BEST_PRACTICES)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
