# ai-service RAG 模块操作文档

本文档配合 `ai-service/` 下的 RAG 模块使用。截至 P6 已实现完整工业级 RAG 流水线。

## 1. RAG 在本项目中的定位

- **作用范围**：仅用于"陪伴聊天"场景（`POST /internal/v1/chat/stream`）。
- **作用方式**：对用户最新一条消息执行**完整 RAG 流水线**（危机词兜底 → query 改写 + HyDE → 多路召回 + RRF 融合 → rerank），从本地心理健康知识库里取 top-k 相关片段，拼接到 system prompt 末尾，再交给 DeepSeek 生成回复。
- **不涉及**：
  - 日记 AI 分析（`/internal/v1/mood/analyze`）保持原样，不注入检索内容。
  - backend（Spring Boot）不做任何改动。
  - 用户自身的日记、聊天记录**不会**被放入知识库，避免跨用户隐私交叉。
- **边界**：本模块只是"知识增强"的陪伴对话，不是医疗诊断工具；知识库里同样写明了危机求助渠道。

## 1.1 整体架构

```
用户消息(可能模糊) ──► 危机词检测 ─────────► 直接注入 crisis-resources（强制兜底）
                       │
                       ├──► multi-query 改写（DeepSeek）
                       └──► HyDE 假答案（仅极短 query）
                                │
                                ▼
              每个 query × { Dense（Qdrant + 千问 embedding）  ┐
                              Sparse（BM25 + jieba） ─────────┘ }
                                │
                                ▼
                            RRF 融合（k=60）→ 候选 20~30
                                │
                                ▼
                       Qwen3-Reranker-0.6B 精排 → top 3~5
                                │
                                ▼
                    拼到 system prompt → DeepSeek 流式生成
```

## 2. 目录结构

```
ai-service/
  knowledge/
    docs/                              仓库内维护的原始知识 Markdown（会被提交）
      # —— 基础理论与方法 ——
      cbt-basics.md                    认知行为疗法（CBT）基础
      acceptance-act-basics.md         接纳承诺疗法（ACT）基础
      dbt-skills-overview.md           辩证行为疗法（DBT）核心技能
      mindfulness-meditation.md        正念冥想入门
      self-compassion.md               自我关怀
      positive-psychology.md           积极心理学与幸福感建设
      meaning-and-purpose.md           意义感与人生方向
      # —— 情绪与症状自助 ——
      anxiety-coping.md                焦虑应对
      panic-attacks.md                 惊恐发作完整应对
      depression-self-help.md          抑郁情绪自助
      seasonal-mood.md                 季节性情绪与冬季低落
      emotion-regulation.md            情绪调节技能
      anger-management.md              愤怒管理
      shame-and-guilt.md               羞耻与内疚
      grief-and-loss.md                悲伤与丧失
      loneliness-coping.md             孤独感应对
      social-anxiety.md                社交焦虑应对
      health-anxiety.md                健康焦虑与疑病
      ocd-tendencies.md                强迫倾向自助
      news-cycle-anxiety.md            应对新闻焦虑
      # —— 行为与习惯 ——
      breathing-relaxation.md          呼吸与放松技巧
      sleep-hygiene.md                 睡眠卫生与失眠应对
      nightmares.md                    噩梦与梦境困扰
      morning-routine.md               早晨例程
      exercise-and-mood.md             运动与心理健康
      nutrition-and-mood.md            饮食与情绪
      mood-journaling.md               情绪日记方法
      procrastination.md               拖延应对
      perfectionism.md                 完美主义应对
      screen-balance.md                数字平衡与手机使用
      addictive-behaviors.md           行为成瘾自助
      adhd-self-awareness.md           注意缺陷倾向自我认识
      # —— 关系与沟通 ——
      relationship-communication.md    人际沟通与关系健康
      romantic-relationship-health.md  健康亲密关系
      breakup-recovery.md              分手与失恋恢复
      family-conflict.md               家庭关系与冲突
      holiday-and-family-gathering.md  节假日与家庭聚会
      friendship-quality.md            友谊的维护与质量
      saying-no-skills.md              学会拒绝
      assertiveness.md                 坚定表达
      gaslighting-recognition.md       识别 PUA 与 gaslighting
      forgiveness-and-letting-go.md    宽恕与放下
      helping-friend.md                如何陪伴有困扰的朋友
      caregiver-burden.md              照顾者的疲惫
      # —— 学业 / 工作 / 财务场景 ——
      stress-management.md             压力管理
      burnout-recovery.md              倦怠识别与恢复
      academic-stress.md               学业与考试压力
      workplace-mental-health.md       职场心理健康
      workplace-bullying.md            职场霸凌与系统性消耗
      financial-anxiety.md             财务焦虑应对
      impostor-syndrome.md             冒充者综合征
      comparison-trap.md               比较陷阱与社交媒体焦虑
      life-transitions.md              重大人生过渡期
      # —— 自我与身份 ——
      self-esteem.md                   自尊与自我价值感
      body-image-acceptance.md         身体形象与自我接纳
      highly-sensitive-person.md       高敏感人群（HSP）
      inner-child-and-childhood.md     童年经历对成年的影响
      # —— 创伤、慢病与专业资源 ——
      trauma-recovery-basics.md        创伤与恢复基础
      chronic-illness-and-mood.md      慢性疾病与情绪
      therapy-first-session.md         心理咨询初体验
      psych-medication-basics.md       关于精神科药物的常见认知
      crisis-resources.md              求助渠道与危机边界
    .cache/                            自动生成的索引缓存（已被 .gitignore 忽略）
      index.faiss / vectors.npy / meta.json   FAISS / numpy 兜底缓存
      qdrant_meta.json                 Qdrant 写入指纹（避免每次启动都重嵌）
      bm25_index.pkl                   BM25 序列化索引
    .qdrant/                           Qdrant 嵌入式模式存储（无 docker 时使用）
  docker-compose.yml                   Qdrant 一键启动
  rag.py                               Embedder 与 KnowledgeRetriever（编排向量化）
  vector_stores.py                     VectorStore 抽象（FAISS / Qdrant）+ 知识分类映射
  bm25_index.py                        BM25 稀疏索引（jieba + rank-bm25）
  rerankers.py                         重排序（Qwen3-Reranker / Noop）
  query_rewriter.py                    危机词检测 + multi-query + HyDE
  pipeline.py                          完整流水线编排（hybrid + rerank + rewrite + crisis）
  eval_rag.py                          离线评估脚本
  eval/qa_dataset.json                 标注 QA 评估集
  main.py                              FastAPI 主入口（已接入 pipeline）
```

> 当前知识库共 **62 篇**、约 117 KB 中文文本，按默认 `RAG_CHUNK_SIZE=400 / RAG_CHUNK_OVERLAP=60` 切分后约 **400 段**。覆盖基础疗法、情绪与症状、行为习惯、关系沟通、学业 / 工作 / 财务、自我与身份、创伤 / 慢病 / 专业资源 七大类，足以让 RAG 检索召回多样化片段，避免"老命中同几个文件"。

## 3. 依赖安装

在 `ai-service` 目录执行：

```powershell
py -m pip install -r requirements.txt
```

依赖一览：

| 包 | 用途 | 何时必需 |
| --- | --- | --- |
| `openai` | DeepSeek 聊天 + DashScope embedding 兼容协议 | 始终需要 |
| `numpy` | 向量运算 | 始终需要 |
| `qdrant-client` | Qdrant 向量库客户端（远程或嵌入式） | `RAG_VECTOR_STORE=qdrant` 时（默认） |
| `faiss-cpu` | FAISS 兜底向量库 | `RAG_VECTOR_STORE=faiss` 时；Windows 安装失败可跳过 |
| `sentence-transformers` | 本地 embedder（千问 Qwen3-Embedding / BGE 等） | `RAG_EMBEDDER=qwen-local` 或 `local` 时（默认） |
| `transformers` + `torch` | 本地 reranker（Qwen3-Reranker） | `RAG_RERANKER=qwen-local` 时（默认） |
| `rank-bm25` + `jieba` | 多路召回（BM25 + 中文分词） | `RAG_ENABLE_BM25=true` 时（默认） |

> 走默认本地千问方案（`qwen-local` embedder + `qwen-local` reranker）需要约 **2.5GB 内存**：
> Qwen3-Embedding-0.6B ~1.2GB + Qwen3-Reranker-0.6B ~1.2GB（fp16 / auto 时）。
> 资源紧张可以：
>  - 把 `RAG_EMBEDDER=dashscope` 走远程，降到 < 100MB；
>  - 或 `RAG_RERANKER=noop` / `RAG_ENABLE_RERANK=false` 关掉重排序。
>
> Windows 下 `faiss-cpu` 安装失败可以暂时跳过：代码内置 numpy 余弦相似度兜底；走 Qdrant 时 `faiss-cpu` 也完全不需要。

### 3.1 Qdrant 部署方式（推荐 docker）

仓库根目录 `ai-service/docker-compose.yml` 已经写好 Qdrant 配置，启动一行：

```powershell
cd ai-service
docker compose up -d qdrant
# Web UI: http://localhost:6333/dashboard
```

如果暂时没有 docker，可以走**嵌入式模式**（零 Docker）：

```env
QDRANT_PATH=knowledge/.qdrant
```

设置 `QDRANT_PATH` 后会优先使用本地文件存储（单进程独占目录），仍可使用 payload 过滤等特性。

> **重要**：嵌入式存储**不能与 `uvicorn --reload` 同时使用**。`--reload` 会启动监视进程 + 工作进程，二者都会创建 `QdrantClient(path=...)`，第二个进程会报错：`Storage folder ... is already accessed by another instance`。处理方式：**改用 Docker + `QDRANT_URL`**；或本地嵌入式时启动不加 `--reload`（例如 `py -m uvicorn main:app --host 0.0.0.0 --port 8001`）。

## 4. 首次启动流程

1. 准备 `.env`（已在 `.env.example` 中追加 RAG 相关配置），默认全部开启即可。
2. 启动 ai-service：

   ```powershell
   cd ai-service
   py -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
   ```

3. **首次启动**会发生：
   - 默认配置下：调用 DashScope 嵌入全部知识片段（首次约 10 万中文 token，<1 元）；写入 Qdrant `mental_health_knowledge` collection。
   - 若 `RAG_EMBEDDER=local`：sentence-transformers 从 HuggingFace 下载 `BAAI/bge-small-zh-v1.5`（约 100MB）。
   - 若 `RAG_VECTOR_STORE=faiss`：将向量 + 元信息落盘到 `knowledge/.cache/`（`vectors.npy` / `meta.json` / `index.faiss`）。
   - 日志里会出现：`RAG 索引已重建：store=qdrant(...)，共 N 段`。
4. **第二次启动**会直接复用已有 store，秒级启动。只有当 `docs/` 里的文件内容、embedder、store 等任一项变化时，才会自动重建。

> 如果下载 HuggingFace 很慢或失败，可在 `.env` 中取消注释 `HF_ENDPOINT=https://hf-mirror.com`。

## 5. 如何新增 / 修改知识

1. 在 `ai-service/knowledge/docs/` 下新增或修改 `*.md` 文件。建议：
   - 每篇聚焦一个主题，文件名用 `kebab-case`。
   - 文首给一个 `# 标题`，内容用段落和列表组织，便于切分。
   - 不要加入任何用户真实信息。
2. 重建索引有两种方式：
   - **自动**：直接重启 ai-service，程序会检测到文件 mtime 变化后重建。
   - **手动**：保持服务运行，调用 `POST /internal/v1/rag/reindex`（见第 7 节）。

## 6. 环境变量一览

### 6.1 通用 RAG 参数

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_ENABLED` | `true` | 关闭时完全降级，走老的 system prompt |
| `RAG_TOP_K` | `3` | 每次检索返回的片段数（建议 2~4） |
| `RAG_MIN_SCORE` | `0.35` | 余弦相似度阈值，低于该值的片段不注入 |
| `RAG_CHUNK_SIZE` | `400` | 每段最大字符数 |
| `RAG_CHUNK_OVERLAP` | `60` | 相邻段的重叠字符数 |
| `RAG_DOCS_DIR` | `knowledge/docs` | 知识文件目录（相对 ai-service 启动目录） |
| `RAG_CACHE_DIR` | `knowledge/.cache` | 索引缓存目录 |
| `RAG_QUERY_CACHE_SIZE` | `256` | 用户查询向量的内存 LRU 缓存条数（设为 0 关闭，避免相同问句重复打远程接口） |

### 6.2 Embedder 选择

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_EMBEDDER` | `qwen-local` | `qwen-local` 本地千问；`dashscope` 远程 DashScope；`local` 通用 sentence-transformers |
| `RAG_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | **仅 `RAG_EMBEDDER=local` 时生效** |

#### 千问本地 embedding（仅 `RAG_EMBEDDER=qwen-local` 时生效）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `QWEN_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 0.6B fp16 约 1.2GB；GPU 充足可换 4B / 8B |
| `QWEN_EMBED_QUERY_PROMPT_NAME` | `query` | Qwen3-Embedding 推荐 query 端走 prompt（doc 不传） |
| `QWEN_EMBED_DEVICE` | _(空，自动)_ | `cuda` / `cpu` / `mps`；空 = 自动检测 |
| `QWEN_EMBED_DTYPE` | `auto` | `auto` / `float16` / `bfloat16` / `float32` |
| `QWEN_EMBED_BATCH_SIZE` | `8` | 文档批量嵌入大小 |

### 6.3 DashScope（阿里云）embedding 配置（仅 `RAG_EMBEDDER=dashscope` 时生效）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | _(空)_ | 必填，[控制台获取](https://dashscope.console.aliyun.com/apiKey) |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容端点，不用改 |
| `DASHSCOPE_EMBED_MODEL` | `text-embedding-v3` | 推荐 v3（1024 维、中文 SOTA）；可选 v2 / v1 |
| `DASHSCOPE_EMBED_DIM` | `1024` | v3 默认 1024；服务端不支持改维度时会按实际返回 |
| `DASHSCOPE_TIMEOUT_SECONDS` | `30` | 单次请求超时 |

> **切换 embedder 不需要手动清缓存**：`KnowledgeRetriever` 的指纹中包含 `embedder.identifier`，
> 一旦 `RAG_EMBEDDER` / `DASHSCOPE_EMBED_MODEL` / `DASHSCOPE_EMBED_DIM` 任一发生变化，
> 下次启动会自动重建索引。

### 6.4 向量库（VectorStore）配置（P2 新增）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_VECTOR_STORE` | `qdrant` | `qdrant` / `faiss`，前者初始化失败会自动回退到 faiss |
| `QDRANT_URL` | `http://localhost:6333` | 远程模式 URL；嵌入式模式下被忽略 |
| `QDRANT_PATH` | _(空)_ | **嵌入式**：本地目录（如 `knowledge/.qdrant`），优先级高于 URL；**勿与 `uvicorn --reload` 同用** |
| `QDRANT_API_KEY` | _(空)_ | Qdrant Cloud 或自建鉴权时填写 |
| `QDRANT_COLLECTION` | `mental_health_knowledge` | collection 名 |
| `QDRANT_PREFER_GRPC` | `false` | true 时走 6334 gRPC（一般不需要） |
| `QDRANT_TIMEOUT_SECONDS` | `30` | 单次请求超时 |

> **payload 携带 category**：每个片段写入 Qdrant 时会附带 `source / title / content / category` 四个字段。
> `category` 来自 `vector_stores.infer_category()` 的静态映射，覆盖 7 大主题，未匹配文件名会落到 `general`。
> 在 `pipeline.search(category="emotions")` 中传入 category 即可启用"按主题条件检索"。

### 6.5 多路召回（P3）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_ENABLE_BM25` | `true` | 关闭后只走 dense |
| `RAG_CANDIDATE_POOL` | `20` | 每路召回拉多少候选（用于 RRF 融合的池子） |
| `RAG_RRF_K` | `60` | RRF 平滑常量，经验取 60 |

> BM25 用 `jieba.cut_for_search` + 单字增强分词，对中文短查询和专有名词（PUA / SSRI / MBTI 等）召回稳定。

### 6.6 重排序（P4）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_RERANKER` | `qwen-local` | `qwen-local` 本地 Qwen3-Reranker；`noop` 不重排 |
| `RAG_ENABLE_RERANK` | `true` | 总开关，关掉等同 noop |
| `RAG_RERANK_INPUT_SIZE` | `20` | 送入 rerank 的候选数 |
| `QWEN_RERANK_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | 也可用 4B / 8B |
| `QWEN_RERANK_DEVICE` | _(空，自动)_ | `cuda` / `cpu` |
| `QWEN_RERANK_DTYPE` | `auto` | 同 embed |
| `QWEN_RERANK_MAX_LENGTH` | `4096` | 单条 (query, doc) 拼接后的最大 token 数 |
| `QWEN_RERANK_BATCH_SIZE` | `8` | 一次送多少 (query, doc) 对 |
| `QWEN_RERANK_INSTRUCTION` | _(空)_ | 自定义 rerank instruction，留空走默认中英文心理健康 prompt |

> Qwen3-Reranker 是 causal LM，按官方 README 用 yes/no token 概率打分。
> 0.6B 在 CPU 上 rerank 20 个候选约 2~6 秒；GPU 上 100~300ms。

### 6.7 查询改写（P5）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_ENABLE_QUERY_REWRITE` | `true` | 总开关；关掉 = 直接用原始 query 检索 |
| `RAG_ENABLE_MULTI_QUERY` | `true` | 让 DeepSeek 把原 query 改写成 2~3 条更适合检索的版本 |
| `RAG_ENABLE_HYDE` | `true` | 让 DeepSeek 写假答案，用假答案的 embedding 去检索 |
| `RAG_MAX_REWRITES` | `3` | 改写后查询数上限（含原 query） |
| `RAG_HYDE_MIN_QUERY_CHARS` | `10` | 仅当 query 长度 ≤ 该值时启用 HyDE（短 query 才需要） |
| `RAG_CRISIS_FORCE_INJECT` | `true` | 命中危机词时直接注入 `crisis-resources.md`，不再走相似度检索 |
| `RAG_CRISIS_TOP_K` | `2` | 危机兜底返回的 crisis-resources 段数 |

调参建议：

- 命中太少 → 调低 `RAG_MIN_SCORE` 到 0.25 / 提高 `RAG_CANDIDATE_POOL` 到 30。
- 命中不相关 → 提高 `RAG_MIN_SCORE` 到 0.4 / 减少 `RAG_TOP_K`；确认 `RAG_ENABLE_RERANK=true`。
- rerank 太慢 → 减小 `RAG_RERANK_INPUT_SIZE`（如 10）/ 关闭改写减少 query 数 / 把 rerank 切到远程或 noop。
- 改写带噪 → 关闭 `RAG_ENABLE_HYDE`（HyDE 对长 query 有时反而拉偏）。

## 6.8 评估脚本（P6）

```powershell
cd ai-service
py eval_rag.py                                 # 跑全量评估，按当前 .env 配置
py eval_rag.py --no-rerank --no-rewrite        # 消融对比：关掉 rerank / 改写
py eval_rag.py --no-bm25                       # 只用 dense
py eval_rag.py --output eval/results-full.json # 把逐条结果存盘
py eval_rag.py --limit 5 --verbose             # 调试用：只跑前 5 条 + INFO 日志
```

输出：
```
配置：embedder=qwen-local, store=qdrant, reranker=qwen-local, bm25=on, rerank=on, rewrite=on
...
总体指标：
  count    = 25
  hit@1    = 0.84
  hit@3    = 0.96
  hit@5    = 1.00
  mrr      = 0.91
  avg_ms   = 850.0
```

> 评估集 `eval/qa_dataset.json` 共 25 条，覆盖 7 大主题 + 1 条危机词测试。
> 如果想加自己的测试用例，直接在 JSON 里追加即可。

## 7. 如何验证 RAG 是否生效

### 7.1 观察启动日志

启动成功后应看到类似：

```
RAG 初始化完成，知识片段数：约 400
```

> 62 篇文档按 `RAG_CHUNK_SIZE=400`、`RAG_CHUNK_OVERLAP=60` 切分后，片段数当前约 **400 段**（每个片段约 250~400 字符，相邻片段重叠 60 字符），属于正常范围。
> 如果你后续修改了 `docs/`，可以临时跑一段简单脚本验证切分数量（参考 `rag.py` 中 `_chunk_text` 方法），再决定是否需要调整 chunk 大小。

如果看到 `RAG 初始化失败` 或 `RAG 知识库目录为空`，请检查 `knowledge/docs/` 路径与文件。

### 7.2 直接调用检索接口

```powershell
curl -X POST http://localhost:8001/internal/v1/rag/search `
     -H "Content-Type: application/json" `
     -d '{"query":"我最近总是睡不着怎么办"}'
```

返回的 `hits[0].source` 应接近 `sleep-hygiene.md`，`score` 通常在 0.4~0.7 之间。

如果配置了 `AI_SERVICE_INTERNAL_TOKEN`，请加上 `-H "X-Internal-Token: xxx"`。

### 7.3 聊天场景观察

发起一次聊天（通过 backend 或直接调 `/internal/v1/chat/stream`），在用户消息里包含关键词（如"焦虑"、"失眠"、"压力"），观察 ai-service 日志：

```
RAG 命中 3 个片段：['sleep-hygiene.md', 'breathing-relaxation.md', 'anxiety-coping.md']
```

回复内容通常会更具体（给出呼吸法、睡眠卫生等可执行建议），而不是泛泛共情。

### 7.4 手动重建索引

修改了 `docs/` 下文件却不想重启服务：

```powershell
curl -X POST http://localhost:8001/internal/v1/rag/reindex
```

返回体里的 `chunkCount` 表示新索引的片段数。

## 8. 关闭 RAG

若需要对比 RAG 前后的效果，或排查问题：

```env
RAG_ENABLED=false
```

重启后 `get_retriever` 会直接返回 `None`，聊天流程走原先的 system prompt，完全兼容旧逻辑。

## 9. 常见问题

### 9.1 模型下载卡住 / 失败

- 现象：首次启动长时间卡在 "正在加载 embedding 模型" 或报 `ConnectionError`。
- 处理：
  1. 在 `.env` 里加 `HF_ENDPOINT=https://hf-mirror.com`。
  2. 或手动预下载模型到本地，然后把 `RAG_MODEL_NAME` 指向本地目录绝对路径。

### 9.2 `faiss-cpu` 安装失败（Windows 常见）

- 处理：先跳过它，`rag.py` 在 `import faiss` 失败时会自动使用 numpy 余弦相似度，不影响功能。等有时间再 `pip install faiss-cpu` 即可。

### 9.3 RAG 注入了无关内容

- 把 `RAG_MIN_SCORE` 调高、`RAG_TOP_K` 调小，再观察。
- 也可以在 `knowledge/docs/` 补充更精细的主题文档，让不同主题拉开向量距离。

### 9.4 索引目录变大 / 想重置

- 直接删除 `ai-service/knowledge/.cache/` 目录，下一次启动会重建。

### 9.5 进程内存占用高

- `bge-small-zh-v1.5` 加载后常驻约 300~500MB。使用 `reload` 模式的 uvicorn 在文件改动时会重新加载，首次加载之后内存会释放再分配一次，属于正常现象。

## 10. 升级路线（实施进度）

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| **P1** | 抽 `Embedder` 接口；接入 DashScope / 本地 / 千问；查询向量 LRU 缓存 | ✅ 已完成 |
| **P2** | 抽 `VectorStore` 接口；接入 Qdrant（远程 / 嵌入式）；payload 携带 category | ✅ 已完成 |
| **P3** | 多路召回（dense + BM25/jieba）+ RRF 融合 | ✅ 已完成 |
| **P4** | 接入 Qwen3-Reranker-0.6B 本地重排序 + Noop 兜底 | ✅ 已完成 |
| **P5** | Query 改写 + HyDE + 危机词强制兜底 | ✅ 已完成 |
| **P6** | `eval_rag.py` 标注 QA 评估脚本（hit@k / MRR） | ✅ 已完成 |

其他未列入路线、可选的扩展：

- 为日记分析接口接入 RAG（需要解决"不让 AI 直接复述知识"的提示词工程）。
- 将知识文件改为从数据库 / 管理后台维护，方便非技术同学贡献内容。
