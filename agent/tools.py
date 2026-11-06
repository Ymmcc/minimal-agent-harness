"""Tool registration, JSON Schema validation and deliberately bounded tools."""
import ast
import math
import operator
from dataclasses import dataclass

from jsonschema import Draft202012Validator, ValidationError


@dataclass
class ToolContext:
    store: object
    user: str
    sid: str
    owner: str
    call_id: str


class Registry:
    def __init__(self):
        self.tools = {}

    def register(self, name, description, schema, handler):
        if name in self.tools:
            raise ValueError(f"工具重复注册: {name}")
        Draft202012Validator.check_schema(schema)
        self.tools[name] = (description, schema, handler)

    def schemas(self):
        return [{"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}
                for name, (desc, schema, _) in self.tools.items()]

    def execute(self, name, arguments, context):
        if name not in self.tools:
            return {"ok": False, "error": "unknown_tool"}
        _, schema, handler = self.tools[name]
        try:
            Draft202012Validator(schema).validate(arguments)
            return {"ok": True, "data": handler(context, **arguments)}
        except ValidationError:
            return {"ok": False, "error": "invalid_arguments", "message": "参数不符合工具 Schema，请检查必填字段和类型。"}
        except (ValueError, ArithmeticError, SyntaxError) as exc:
            return {"ok": False, "error": "tool_error", "message": str(exc)[:200]}
        except Exception:
            # Don't expose stack traces / credentials from unexpected handlers.
            return {"ok": False, "error": "tool_failed", "message": "工具内部错误，本次调用失败。"}


def calculator(_context, expression):
    """Allow only short numeric arithmetic. Never evaluate Python code."""
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 50:
        raise ValueError("表达式过于复杂")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv,
                  ast.Mod: operator.mod}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("仅支持数字、括号和 + - * / %")
        if abs(value) > 1e12 or not math.isfinite(value):
            raise ValueError("计算结果超出允许范围")
        return value

    return {"value": visit(tree.body)}


def make_registry():
    registry = Registry()

    def schema(properties, required, **extra):
        return {"type": "object", "properties": properties, "required": required,
                "additionalProperties": False, **extra}

    registry.register("calculator", "计算算术表达式，支持 + - * / % 和括号。",
                      schema({"expression": {"type": "string", "minLength": 1, "maxLength": 200}}, ["expression"]), calculator)
    registry.register("search", "离线模拟搜索；结果标注 mock，不提供实时资讯或天气。",
                      schema({"query": {"type": "string", "minLength": 1, "maxLength": 200}}, ["query"]),
                      lambda ctx, query: {"mock": True, "query": query, "results": [
                          {"title": "Agent 笔试说明（本地演示数据）", "snippet": "Agent 通过模型决策、工具执行和结果反馈循环完成任务。"}]
                      })

    def todo(ctx, action, text=None, item_id=None):
        if action == "add" and not text.strip():
            raise ValueError("待办内容不能为空白")
        return ctx.store.todo(ctx.user, ctx.sid, ctx.owner, ctx.call_id, action, text, item_id)

    todo_schema = schema({
        "action": {"type": "string", "enum": ["add", "list", "done"]},
        "text": {"type": "string", "minLength": 1, "maxLength": 300},
        "item_id": {"type": "integer", "minimum": 1},
    }, ["action"], allOf=[
        {"if": {"properties": {"action": {"const": "add"}}}, "then": {"required": ["text"]}},
        {"if": {"properties": {"action": {"const": "done"}}}, "then": {"required": ["item_id"]}},
    ])
    registry.register("todo", "管理当前 session 的待办；add 需要 text，done 需要 item_id，list 列出全部。",
                      todo_schema, todo)
    return registry
