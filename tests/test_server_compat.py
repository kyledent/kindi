"""Client compatibility of the OpenAI server: the thinking switch and efforts, thinking budgets, and the error for a
prompt longer than the context. Needs no GPU: the server runs with its real chat templates, tokenizer and prompt
preparation, and a stub in place of the engine. The Gemma 4 path is not covered.

    python3 tests/test_server_compat.py PACK [PACK ...]     (each a converted pack: the 27B, and a 2B/4B/9B)

In Docker, with the packs in ~/.kindi:
    docker run --rm -v "$PWD":/src -v ~/.kindi:/data -w /src kindi:0.3.0 python3 tests/test_server_compat.py /data/pack
"""
import asyncio
import json
import os
import re
import socket
import sys
import threading
import time
import types
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kindi import chat, config, server                      # noqa: E402
from kindi.engine import EOS                                 # noqa: E402
from kindi.tokenizer import Tok                              # noqa: E402

CTX = 512
# what clients look for in the message: see docs/AGENTS.md
OVERFLOW_PATTERNS = [r"maximum context length is \d+ tokens", r"reduce the length of the messages",
                     r"this model's maximum context length is"]


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print("  ok  " + what)


def unit():
    print("thinking_of / thinking_budget / config")
    cfg = {"reasoning_effort": "", "thinking_budget": 4096, "thinking_budgets": {"low": 1024, "xhigh": 8192}}
    t = server.thinking_of
    check(t({}, cfg) == (False, None), "no switch, no effort: thinking off")
    check(t({"chat_template_kwargs": {"enable_thinking": True}}, cfg) == (True, None), "enable_thinking turns it on")
    check(t({"enable_thinking": True}, cfg) == (True, None), "top-level enable_thinking turns it on")
    check(t({"reasoning_effort": "high"}, cfg) == (True, "xhigh"), "reasoning_effort high alone: on, xhigh")
    check(t({"reasoning_effort": "minimal"}, cfg) == (True, "low"), "minimal maps to low")
    check(t({"reasoning_effort": "MEDIUM"}, cfg) == (True, "medium"), "medium, any case")
    check(t({"reasoning": {"effort": "low"}}, cfg) == (True, "low"), "reasoning.effort (OpenRouter style)")
    check(t({"reasoning_effort": "none"}, cfg) == (False, None), "none: thinking off")
    check(t({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}}, cfg)[0] is False,
          "an explicit enable_thinking false wins over an effort")
    check(t({"chat_template_kwargs": {"enable_thinking": True}}, dict(cfg, reasoning_effort="medium")) ==
          (True, "medium"), "no effort in the request: the server's reasoning_effort")
    b = server.thinking_budget
    check(b({"thinking_budget_tokens": 300}, cfg, True, "xhigh") == 300, "a request's own budget wins")
    check(b({}, cfg, True, "xhigh") == 8192 and b({}, cfg, True, "low") == 1024, "the budget for the effort")
    check(b({}, cfg, True, "medium") == 4096 and b({}, cfg, True, None) == 4096, "an effort not listed: thinking_budget")
    check(b({"thinking_budget_tokens": 300}, cfg, False, "xhigh") == 0, "no budget with thinking off")
    os.environ["KINDI_THINKING_BUDGETS"] = "low:2048, xhigh:16384"
    check(config.load()["thinking_budgets"] == {"low": 2048, "xhigh": 16384}, "KINDI_THINKING_BUDGETS is parsed")
    os.environ["KINDI_THINKING_BUDGETS"] = "high:2048"
    try:
        config.load()
        check(False, "a bad thinking_budgets is refused")
    except SystemExit:
        check(True, "a bad thinking_budgets is refused")
    del os.environ["KINDI_THINKING_BUDGETS"]


def stub_generate(self, req, cancel=None, prep=None):
    """the engine's part, stubbed: report what prepare() made of the request"""
    prep = prep or self.prepare(req)
    time.sleep(float(req.get("_sleep", 0)))
    text = self.tok.decode(prep["T"])
    m = re.search(r"Reasoning effort is set to (\w+)", text)
    content = json.dumps({"think": prep["think"], "budget": prep["budget"], "tokens": len(prep["T"]),
                          "effort_line": m.group(1) if m else None})
    yield "text", content
    yield "done", {"content": content, "tool_calls": [], "reasoning": "", "finish": "stop",
                   "usage": {"prompt_tokens": len(prep["T"]), "completion_tokens": 1, "total_tokens": len(prep["T"]) + 1},
                   "timing": {}}


def stub_kindi(cfg):
    K = server.Kindi.__new__(server.Kindi)
    K.cp, K.cfg, K.log = None, cfg, print
    K.native = False
    K.variant = chat.template_variant(cfg["pack"])
    K.eng = types.SimpleNamespace(ctx=CTX)
    K.tok = Tok(json.load(open(os.path.join(cfg["pack"], "vocab.json"))), os.path.join(cfg["pack"], "merges.txt"))
    K.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0, "read": 0, "generated": 0, "started": 0.0}
    K.lock = threading.Lock()
    K.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
    K.t0 = time.time()
    K.generate = types.MethodType(stub_generate, K)
    return K


def run_script(K, req, text, prep=None):
    """Kindi.generate itself, with a model that "writes" text then stops: (reasoning, answer, info)"""
    tokens = iter(K.tok.encode(text) + [sorted(EOS)[0]])
    K.sample = lambda *a: next(tokens)
    out = {"reasoning": "", "text": ""}
    info = None
    for kind, val in K.generate(req, None, prep):
        if kind == "done":
            info = val
        else:
            out[kind] += val
    return out["reasoning"], out["text"], info


def real_generate(pack):
    """the real generate loop and prompt preparation, with a scripted model"""
    os.environ["KINDI_PACK"] = pack
    cfg = config.load()
    cfg["thinking_budget"], cfg["thinking_budgets"] = 0, {"low": 8}
    logged = []
    K = stub_kindi(cfg)
    del K.generate
    K.log = logged.append
    K.eng = types.SimpleNamespace(ctx=CTX, hsl=None, shortlist=0)
    K.runner = types.SimpleNamespace(prepare=lambda T, body_len, end, cancel: (None, 0), step=lambda tok: None)
    print("generate, %s" % pack)
    hi = [{"role": "user", "content": "Say hi."}]

    r, a, info = run_script(K, {"messages": hi}, "Hi there.")
    check(r == "" and a == "Hi there." and info["finish"] == "stop", "thinking off: only an answer")
    r, a, _ = run_script(K, {"messages": hi, "reasoning_effort": "high"}, "A plan.</think>\n\nHi there.")
    check(r.strip() == "A plan." and a == "Hi there.", "reasoning_effort alone: thinking comes back apart")

    long_thought = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    r, a, _ = run_script(K, {"messages": hi, "reasoning_effort": "low"}, long_thought + "</think>\n\nDone.")
    check(any("thinking budget of 8 tokens" in m for m in logged), "the server's budget for the effort closes the thinking")
    check("one" in r and a != "" and "fifteen" not in r, "after the budget the model's text is the answer")
    del logged[:]
    r, a, _ = run_script(K, {"messages": hi, "reasoning_effort": "low", "thinking_budget_tokens": 500},
                         long_thought + "</think>\n\nDone.")
    check(not logged or not any("thinking budget" in m for m in logged), "a request's own budget wins")
    check("fifteen" in r and a == "Done.", "within its budget the model finishes its thinking")

    prep = K.prepare({"messages": hi})
    K.prepare = None                                         # generate must not prepare again
    r, a, _ = run_script(K, {"messages": hi}, "Hi.", prep)
    check(a == "Hi.", "generate uses the prepared prompt")
    del K.prepare
    try:
        list(K.generate({"messages": [{"role": "user", "content": "word " * (CTX * 2)}]}))
        check(False, "generate without prep refuses a prompt that is too long (kindi chat)")
    except OverflowError as e:
        check("maximum context length" in str(e), "generate without prep refuses a prompt that is too long (kindi chat)")


def start(cfg):
    import uvicorn
    server._make_engine = lambda c, log: stub_kindi(c)
    app = server.create_app(cfg, log=lambda m: None)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    srv.install_signal_handlers = lambda: None

    def run():
        asyncio.set_event_loop(asyncio.new_event_loop())    # uvicorn on Python 3.6 uses the thread's current loop
        srv.run()
    threading.Thread(target=run, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1)
            return "http://127.0.0.1:%d" % port, srv
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("the test server did not start")


def post(base, body):
    """(status, content type, body text)"""
    r = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.status, f.headers.get("content-type", ""), f.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type", ""), e.read().decode()


def stream_content(text):
    out = ""
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            d = json.loads(line[6:])
            out += d["choices"][0]["delta"].get("content") or ""
    return json.loads(out)


def http(pack):
    os.environ["KINDI_PACK"] = pack
    cfg = config.load()
    cfg["thinking_budget"], cfg["thinking_budgets"] = 4096, {"low": 1024, "medium": 2048, "xhigh": 8192}
    base, srv = start(cfg)
    big27 = chat.template_variant(pack) == "bonsai27"
    print("HTTP, %s (%s template)" % (pack, "27B" if big27 else "2B/4B/9B"))
    hi = [{"role": "user", "content": "Say hi."}]
    long_msgs = [{"role": "user", "content": "word " * (CTX * 2)}]

    m = json.loads(urllib.request.urlopen(base + "/v1/models").read().decode())["data"][0]
    check(m["context_length"] == CTX and m["max_model_len"] == CTX, "/v1/models: context_length and max_model_len")

    for stream in (False, True):
        st, ct, body = post(base, {"messages": long_msgs, "stream": stream})
        e = json.loads(body)["error"] if st == 400 and "json" in ct else {}
        check(st == 400 and e.get("code") == "context_length_exceeded" and e.get("type") == "invalid_request_error"
              and e.get("param") == "messages", "overflow, stream=%s: HTTP 400 context_length_exceeded" % stream)
        check(all(re.search(p, e["message"], re.I) for p in OVERFLOW_PATTERNS),
              "overflow, stream=%s: the message has OpenAI's wording" % stream)

    st, ct, body = post(base, {"messages": hi, "stream": True,
                               "chat_template_kwargs": {"enable_thinking": True}})
    r = stream_content(body)
    check(st == 200 and "event-stream" in ct and r["think"], "a normal request streams")
    check(r["budget"] == (8192 if big27 else 4096),
          "enable_thinking, no effort: budget %d (%s)" % (r["budget"], "the 27B default xhigh" if big27 else "thinking_budget"))
    st, _, body = post(base, {"messages": hi, "reasoning_effort": "high"})
    r = json.loads(json.loads(body)["choices"][0]["message"]["content"])
    check(st == 200 and r["think"] and r["budget"] == 8192, "reasoning_effort high alone: thinking on, xhigh budget")
    if big27:
        check(r["effort_line"] == "xhigh", "the 27B prompt says xhigh")
    st, _, body = post(base, {"messages": hi, "reasoning_effort": "low"})
    r = json.loads(json.loads(body)["choices"][0]["message"]["content"])
    check(r["think"] and r["budget"] == 1024 and (r["effort_line"] == "low" or not big27), "reasoning_effort low")
    st, _, body = post(base, {"messages": hi, "reasoning_effort": "none"})
    r = json.loads(json.loads(body)["choices"][0]["message"]["content"])
    check(not r["think"] and r["budget"] == 0, "reasoning_effort none: thinking off")
    if big27:
        st, _, body = post(base, {"messages": hi, "reasoning_effort": "ultra"})
        check(st == 400 and "reasoning effort" in json.loads(body)["error"]["message"].lower(),
              "an unknown effort is a 400")

    # a request that waits behind another gets its overflow error at once
    slow = threading.Thread(target=post, args=(base, {"messages": hi, "_sleep": 4}))
    slow.start()
    time.sleep(0.5)
    t0 = time.time()
    st, _, _ = post(base, {"messages": long_msgs, "stream": True})
    dt = time.time() - t0
    slow.join()
    check(st == 400 and dt < 2, "overflow while another request runs: 400 after %.2f s, without waiting" % dt)
    srv.should_exit = True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    unit()
    for p in sys.argv[1:]:
        real_generate(p)
        http(p)
    print("ALL PASS")
