"""Local Flask UI; chat and code generation are always performed by the local Transformer."""
import ast, io, contextlib
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from generate import generate
from chat import chat

app=Flask(__name__,template_folder="templates",static_folder="static")
CORS(app, origins=["http://127.0.0.1:5000", "http://localhost:5000"])

@app.get("/")
def home(): return render_template("index.html")

CODE_STARTS = ("def ", "class ", "import ", "from ", "#", "@", "async def ", '"' * 3, "'" * 3)

def as_code_prompt(prompt):
    """The model only learned from raw .py files, so plain English means little to it.
    Turn an English request into a Python comment it can continue; leave code as-is."""
    if "\n" in prompt or prompt.lstrip().startswith(CODE_STARTS): return prompt
    return f"# {prompt}\n"

@app.post("/generate")
def make_code():
    prompt=(request.get_json(silent=True) or {}).get("prompt","").strip()
    if not prompt: return jsonify(error="Enter a prompt first."),400
    try: return jsonify(code=generate(as_code_prompt(prompt),max_new_tokens=200))
    except FileNotFoundError as e: return jsonify(error=str(e)),503
    except Exception as e: app.logger.exception("Generation failed"); return jsonify(error=str(e)),500

@app.post("/chat")
def chat_reply():
    """Body: {"messages": [{"role": "user"|"assistant", "content": "..."}]}. Needs a Stage 2 (chat) model."""
    messages=(request.get_json(silent=True) or {}).get("messages")
    if not isinstance(messages,list) or not messages: return jsonify(error="Send at least one message."),400
    clean=[]
    for m in messages[-20:]:
        if not isinstance(m,dict) or m.get("role") not in ("user","assistant") or not isinstance(m.get("content"),str): return jsonify(error="Each message needs a role (user or assistant) and text."),400
        if len(m["content"])>4000: return jsonify(error="Each message is limited to 4,000 characters."),400
        clean.append({"role":m["role"],"content":m["content"]})
    if clean[-1]["role"]!="user": return jsonify(error="The last message must be from the user."),400
    try: return jsonify(reply=chat(clean,max_new_tokens=200))
    except FileNotFoundError as e: return jsonify(error=str(e)),503
    except Exception as e: app.logger.exception("Chat failed"); return jsonify(error=str(e)),500

@app.post("/run")
def run_code():
    code=(request.get_json(silent=True) or {}).get("code","")
    if not isinstance(code,str) or not code.strip(): return jsonify(error="There is no code to run."),400
    if len(code)>20000: return jsonify(error="Code is limited to 20,000 characters."),400
    # Deliberately tiny language subset. This is a local learning demo, not a secure
    # isolation boundary; never expose this development server to an untrusted network.
    allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Expr,ast.Assign,ast.AnnAssign,ast.Name,ast.Load,ast.Store,ast.Constant,ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,ast.Call,ast.If,ast.For,ast.List,ast.Tuple,ast.Dict,ast.Set,ast.Subscript,ast.Slice,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.USub,ast.UAdd,ast.Not,ast.And,ast.Or,ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.IfExp,ast.keyword,ast.Pass,ast.Break,ast.Continue,ast.ListComp,ast.comprehension)
    try:
        tree=ast.parse(code)
        if any(not isinstance(n,allowed) for n in ast.walk(tree)): return jsonify(error="This demo only runs basic expressions/functions; imports, file access, and other statements are blocked."),400
        def safe_range(*args):
            values=range(*args)
            if len(values)>10000: raise ValueError("range is limited to 10,000 items")
            return values
        env={"__builtins__":{"print":print,"range":safe_range,"len":len,"sum":sum,"min":min,"max":max,"abs":abs,"round":round,"int":int,"float":float,"str":str,"bool":bool,"list":list,"dict":dict,"enumerate":enumerate,"zip":zip}}
        output=io.StringIO()
        with contextlib.redirect_stdout(output): exec(compile(tree,"<generated>","exec"),env,env)
        return jsonify(output=output.getvalue(),defined=[k for k in env if not k.startswith("_")])
    except Exception as e: return jsonify(error=f"{type(e).__name__}: {e}"),400

if __name__=="__main__": app.run(host="127.0.0.1",port=5000,debug=False)
