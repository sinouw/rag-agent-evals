import pytest

from app.tools import CalculatorError, SourceRegistry, calculator, safe_eval


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2023 - 2001)", 22),
        ("14143 / 3", 14143 / 3),
        ("-2 ** 2", -4),
        ("round(10 / 3, 2)", 3.33),
        ("max(1, 7)", 7),
    ],
)
def test_calculator_evaluates_arithmetic(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('ls')", "open('x')", "(1).__class__", "a + 1", "9 ** 9999", "lambda: 1", "1 if 1 else 2"],
)
def test_calculator_rejects_anything_but_arithmetic(expr):
    with pytest.raises(CalculatorError):
        safe_eval(expr)


def test_calculator_tool_reports_errors_to_the_model():
    assert calculator("79 - 72").content == "79 - 72 = 7"
    result = calculator("1 / 0")
    assert result.error and "division by zero" in result.content


def test_sql_tool_allows_select(indexed):
    _, sql, _ = indexed
    result = sql.run("SELECT COUNT(*) AS n FROM documents WHERE status = 'Final';")
    assert not result.error
    assert result.data["rows"] == [[2]]
    assert "CREATE TABLE documents" in sql.schema()


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM documents",
        "UPDATE documents SET title = 'x'",
        "DROP TABLE documents",
        "SELECT 1; DROP TABLE documents",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "WITH x AS (SELECT 1) DELETE FROM documents",
    ],
)
def test_sql_tool_refuses_writes(indexed, statement):
    _, sql, _ = indexed
    result = sql.run(statement)
    assert result.error
    assert sql.run("SELECT COUNT(*) FROM documents").data["rows"] == [[3]]


def test_source_registry_labels_are_stable_across_searches(indexed):
    retriever, _, _ = indexed
    registry = SourceRegistry()
    first = retriever.search("maximum line length", 3)
    labels = [registry.label(h) for h in first]
    assert labels == ["S1", "S2", "S3"]
    assert registry.label(first[0]) == "S1"
    second = retriever.search("assignment expressions walrus", 3)
    assert all(registry.label(h).startswith("S") for h in second)
