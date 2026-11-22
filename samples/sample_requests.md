# Sample API Requests and Responses

Start the server locally:

```bash
uvicorn app.main:app --reload --port 8000
```

## 1. Save Career Goal & Language Preference (Session 1)

**Request:**
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "session_id": "session-456",
    "message": "I'\''m planning to switch jobs next year, and I prefer advice in Hindi."
  }'
```

**Response:**
```json
{
  "user_id": "user-123",
  "session_id": "session-456",
  "response": "आपकी सहेजी गई जानकारी (I'm planning to switch jobs next year) को ध्यान में रखते हुए, अपने मुख्य लक्ष्य के लिए सही समय, कौशल निर्माण और निरंतर प्रयास पर ध्यान दें।",
  "context_used": ["career_goal", "language_preference"],
  "memories_saved": ["career_goal", "language_preference"]
}
```

## 2. Follow-up Question in Same Session

**Request:**
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "session_id": "session-456",
    "message": "Why do you say that?"
  }'
```

**Response:**
```json
{
  "user_id": "user-123",
  "session_id": "session-456",
  "response": "पिछली बात के आधार पर (...), यह सुझाव आपकी वर्तमान स्थिति और प्राथमिकताओं को ध्यान में रखकर दिया गया है।",
  "context_used": ["language_preference"],
  "memories_saved": []
}
```

## 3. Cross-Session Memory Recall (New Session)

**Request:**
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "session_id": "session-789",
    "message": "What do you remember about my career plans?"
  }'
```

**Response:**
```json
{
  "user_id": "user-123",
  "session_id": "session-789",
  "response": "मुझे याद है कि आपने बताया था: I'm planning to switch jobs next year",
  "context_used": ["career_goal", "language_preference"],
  "memories_saved": []
}
```
