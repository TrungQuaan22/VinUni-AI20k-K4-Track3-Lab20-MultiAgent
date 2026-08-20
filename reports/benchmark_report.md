# Multi-Agent Research System — Benchmark & Evaluation Report

## 1. Executive Summary

Báo cáo này đánh giá và so sánh định lượng giữa hai kiến trúc AI Research Assistant:
1. **Single-Agent Baseline**: Mô hình nguyên khối (monolithic) thực hiện trả lời trực tiếp trong 1 lượt prompt duy nhất mà không có công cụ tìm kiếm hoặc phân tích trung gian.
2. **Multi-Agent Workflow**: Hệ thống đa tác tử gồm **Supervisor + Researcher + Analyst + Writer + Critic**, phối hợp thông qua Shared State (`ResearchState`) được điều phối bởi LangGraph StateGraph.

---

## 2. Summary Metrics

| Run | Latency (s) | Cost (USD) | Quality (0-10) | Citation Cov. | Failure Rate | Routing / Notes |
|:---|---:|---:|---:|---:|---:|:---|
| **single_agent** | **10.95s** | **$0.00042** | 8.00/10 | **0.0%** | 0.0% | `single pass (LLM direct)` |
| **multi_agent** | **22.30s** | **$0.00130** | **10.00/10** | **100.0%** | 0.0% | `routes=researcher>analyst>writer>done; sources=5` |

---

## 3. Trade-off Analysis & Key Insights

- **Độ chính xác và Trích dẫn (Citation Rigor & Grounding)**:
  - Single-Agent hoàn toàn không có khả năng tự tìm kiếm và trích dẫn bằng chứng thực tế từ web/corpus ngoài bộ nhớ tham số, dẫn đến **Citation Coverage = 0.0%** và nguy cơ ảo giác cao.
  - Multi-Agent đạt **100.0% Citation Coverage** nhờ Researcher Agent trích xuất nguồn từ Tavily Search / Offline Research Corpus, Analyst phân tích so sánh, và Writer tổng hợp với định dạng chú thích inline `[1]`, `[2]` kèm thư mục tài liệu tham khảo chuẩn xác.
- **Phân tách trách nhiệm (Role Specialization)**:
  - Việc chia nhỏ bài toán thành các vai trò chuyên trách giúp ngăn chặn hiện tượng quá tải context window (*context pollution*), mỗi agent tập trung thực hiện xuất sắc một nhiệm vụ trước khi bàn giao sang bước tiếp theo.
- **Độ trễ và Chi phí (Latency & Cost Trade-off)**:
  - Multi-Agent có độ trễ gấp ~2 lần (22.3s so với 10.95s) và chi phí token gấp ~3.1 lần ($0.00130 so với $0.00042) do nhiều lượt handoff và prompt độc lập. Đây là trade-off hoàn toàn xứng đáng cho các tác vụ nghiên cứu kỹ thuật phức tạp yêu cầu độ tin cậy tuyệt đối.

---

## 4. Failure Modes Analysis & Recovery Strategies

Trong quá trình thiết kế và thực thi hệ thống Multi-Agent, chúng tôi xác định các failure modes chính và các cơ chế phòng vệ (Guardrails) đã triển khai:

| Failure Mode | Nguyên nhân gốc rễ | Hậu quả | Cơ chế khắc phục (Mitigation) |
|---|---|---|---|
| **Runaway Loop (Vòng lặp vô hạn)** | Supervisor không thể chuyển trạng thái hoặc agent liên tục trả lời không đạt yêu cầu | Cạn kiệt budget token, treo workflow | Thiết lập `max_iterations = 6`. Supervisor tự động chuyển trạng thái sang `done` khi đạt ngưỡng tối đa. |
| **Search / LLM API Timeout & Network Error** | Kết nối mạng chập chờn, API bên thứ ba bị nghẽn (Tavily/OpenAI) | Tiến trình bị crash, gián đoạn workflow | Bổ sung retry decorator (`tenacity`) với exponential backoff (tối đa 3 lần) và cơ chế **Graceful Fallback**: `SearchClient` tự động chuyển sang đọc bộ dữ liệu 30 topics `ai_agent_offline_research_corpus_v2`, `LLMClient` tự động chuyển sang heuristic generator. |
| **Context Loss during Handoff** | State truyền giữa các agent bị thiếu thông tin hoặc ghi đè | Worker sau không hiểu ngữ cảnh worker trước | Xây dựng Pydantic schema `ResearchState` chuẩn hoá và thống nhất, lưu vết tuần tự `sources`, `research_notes`, `analysis_notes`, `final_answer`, `route_history`, và `agent_results`. |
| **Hallucinated Citations** | Writer tự tạo ra số trích dẫn không có trong danh mục sources thu thập | Mất uy tín báo cáo khoa học | Analyst chuẩn bị bảng đối chiếu bằng chứng; Writer được ép buộc định dạng theo danh mục `[1]..[N]` có trong state; Critic Agent kiểm tra và chấm điểm tính xác thực. |

---

## 5. Observability & Tracing

Hệ thống được tích hợp sẵn tracing end-to-end:
- **LangSmith Tracing**: Tự động liên kết với project `multi-agent-research-lab` thông qua `LANGCHAIN_TRACING_V2=true` để theo dõi latency từng node, input/output prompt, và token usage thời gian thực.
- **Internal Tracing**: Mỗi node LangGraph và agent invocation được bọc bởi context manager `trace_span`, ghi lại execution duration và metadata vào trường `state.trace` phục vụ debugging độc lập.

---

## 6. Exit Ticket — Khi nào nên và không nên dùng Multi-Agent?

### 1. Case nào nên dùng Multi-Agent? Vì sao?
- **Nghiên cứu kỹ thuật, phân tích dữ liệu chuyên sâu và tổng hợp báo cáo đa nguồn**:
  - *Lý do*: Đòi hỏi phối hợp nhiều kỹ năng riêng biệt (tìm kiếm nguồn, trích xuất dữ kiện, phản biện mâu thuẫn, tổng hợp hành văn có trích dẫn). Khi chia nhỏ thành các Agent chuyên môn hóa, hệ thống tránh được hiện tượng context pollution, cho phép kiểm soát luồng logic chặt chẽ và xác thực chéo (verification) giữa các khâu.

### 2. Case nào không nên dùng Multi-Agent? Vì sao?
- **Tra cứu thông tin đơn giản, định dạng lại văn bản (formatting), dịch thuật trực tiếp, hoặc các câu hỏi một bước (one-turn Q&A)**:
  - *Lý do*: Các tác vụ này không yêu cầu phân tách logic hay xác thực đa vòng. Sử dụng Multi-Agent trong các trường hợp này sẽ gây lãng phí chi phí token không cần thiết, gia tăng độ trễ (latency) đáng kể, và tăng nguy cơ lỗi hệ thống tại các điểm handoff trung gian.
