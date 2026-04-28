# Admin/Audit Feature - Quick Start Guide

## 30-Second Overview
- **What:** Read-only admin interface to inspect KYC sessions and audit logs from PostgreSQL
- **How:** Simple admin key header validation (environment variable based)
- **Where:** Backend: `/admin/sessions`, `/admin/sessions/{id}`, `/admin/sessions/{id}/logs` | Frontend: `/admin`, `/admin/sessions`, `/admin/session/[id]`
- **Access:** No login needed—just enter admin key once per session

---

## Step 1: Set Admin Key (Backend)

### Default Development Setup
```bash
cd /Users/giumi/Documents/Licenta/facial-biometric-authentication

# Terminal 1: Start backend with default dev key
python -m uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000

# Default admin key: "dev-admin-key"
```

### Custom Admin Key
```bash
# Terminal 1: Start backend with custom key
export ADMIN_KEY=your-custom-key-here
python -m uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Step 2: Start Mobile App

```bash
cd visionauth-mobile

# Option A: With Expo CLI
npm start

# Option B: Using the terminal name you already have
# In the "frontend" terminal: npm start
```

---

## Step 3: Access Admin Section

### Option A: Deep Link (Easiest for Testing)
```
1. In Expo dev tools (terminal), press 'w' to open web
2. Look for your phone's tunnel URL
3. Open deep link: exp://YOUR-TUNNEL-URL/admin
4. Or press Ctrl+K in Expo terminal and type: /admin
```

### Option B: Add Test Button to Home Screen
Edit `visionauth-mobile/app/index.tsx` and add before the Start button:

```typescript
<Pressable
  style={styles.adminTestBtn}
  onPress={() => router.push('/admin')}>
  <Text style={styles.adminTestBtnText}>🔧 Admin (dev only)</Text>
</Pressable>
```

Then add this style:
```typescript
adminTestBtn: {
  alignItems: 'center',
  backgroundColor: '#4B5563',
  borderRadius: 8,
  paddingVertical: 12,
  marginBottom: 12,
},
adminTestBtnText: {
  color: '#94A3B8',
  fontSize: 14,
  fontWeight: '600',
},
```

---

## Step 4: Use Admin Interface

### At `/admin` Screen
1. **Enter Admin Key**
   - Input field for the admin key
   - Default for dev: `dev-admin-key`
   - Tap "Access Admin"

2. **Key Validation**
   - ✅ Valid: Routes to sessions list
   - ❌ Invalid: Shows error alert "Invalid admin key"

### At `/admin/sessions` Screen
1. **Browse Sessions**
   - Shows list of KYC sessions (most recent first)
   - Each card shows:
     - Name (or "Unknown")
     - Shortened session ID
     - Status badge (color-coded)
     - Updated timestamp
     - Face match status
     - Liveness status
   
2. **Refresh Sessions**
   - Pull down to refresh list
   - No need to re-enter key

3. **View Details**
   - Tap any session card to see full details

### At `/admin/session/[id]` Screen
1. **Review Complete Data:**
   - **Identity Info:** Name, CNP, Series/Number
   - **Verification Results:** Liveness, face match distance, final decision
   - **Media Paths:** Document, ID face, selfie, liveness video locations
   - **OCR Text:** Full raw text extracted from ID (if available)
   - **Embeddings:** Metadata about stored embeddings (not full vectors)
   - **Audit Log:** Timeline of all events during the session

2. **Go Back**
   - Tap "← Back" to return to sessions list

---

## Step 5: Quick Test Checklist

- [ ] Backend starts without errors
- [ ] Mobile app loads
- [ ] Can navigate to `/admin` screen
- [ ] Entering wrong key shows error
- [ ] Entering correct key (`dev-admin-key`) opens sessions list
- [ ] Sessions list shows any KYC sessions you've run through the app
- [ ] Tapping a session shows full details
- [ ] Audit log shows expected events
- [ ] Can go back to sessions list

---

## Key Points to Remember

### Admin Key Security
- **Default:** `"dev-admin-key"` (for development)
- **Stored:** Backend environment variable `ADMIN_KEY`
- **Passed:** In HTTP header `X-Admin-Key` with each request
- **Protection:** Returns 403 Forbidden if wrong or missing
- **Duration:** Key is stored only in component state (clears on app restart)

### What You Can Do
✅ View all KYC sessions  
✅ See verification results (liveness, face match)  
✅ Read audit logs  
✅ Check stored file paths  
✅ View OCR results  
✅ See embedding metadata (safe preview, not full vectors)  

### What You Cannot Do
❌ Modify sessions  
❌ Delete data  
❌ Create new sessions  
❌ View full embedding vectors (only preview)  
❌ Manage users/roles (no authentication system)  

---

## Endpoints Reference

### Backend Endpoints

```
GET /admin/sessions?limit=50&offset=0
├─ Header: X-Admin-Key: {your-key}
└─ Response: List of recent sessions

GET /admin/sessions/{session_id}
├─ Header: X-Admin-Key: {your-key}
└─ Response: Full session details + embeddings + media paths

GET /admin/sessions/{session_id}/logs
├─ Header: X-Admin-Key: {your-key}
└─ Response: Chronological audit log for session
```

### Frontend Routes

```
/admin
├─ Admin key entry screen
└─ Validates key, navigates to sessions list

/admin/sessions
├─ List of recent KYC sessions
└─ Tap to view session details

/admin/session/[id]
├─ Complete session information
├─ Identity, verification results, media paths
├─ OCR text, embeddings, audit log
└─ Back button to return to list
```

---

## Testing with Real Data

### Generate Test Session
1. Use mobile app normally to complete a full KYC flow:
   - Start verification
   - Upload ID document
   - Review OCR fields
   - Take selfie
   - Pass liveness check
   - Complete face matching

2. This creates a session in PostgreSQL
3. Check admin section—your new session appears in the list

### Check Database Directly
```bash
# Connect to PostgreSQL
psql postgresql://postgres:postgres@localhost:5432/visionauth

# View sessions
select id, first_name, last_name, status, final_decision, created_at 
from kyc_sessions 
order by created_at desc;

# View audit logs for a session
select event_type, message, created_at 
from audit_logs 
where session_id = '{your-session-uuid}'
order by created_at;
```

---

## Troubleshooting

### "Invalid admin key" Error
```
Problem: Can't access admin section
Solution: Verify ADMIN_KEY environment variable matches what you entered
  - Default: dev-admin-key
  - Custom: Check terminal where backend is running
```

### Sessions List is Empty
```
Problem: No sessions showing in /admin/sessions
Solution: You haven't run a complete KYC flow yet
  - Complete a full verification flow in the app
  - Sessions appear in admin section after first flow
```

### Routes Not Found
```
Problem: /admin route doesn't exist
Solution: Restart Expo
  - Stop: Ctrl+C in frontend terminal
  - Restart: npm start
  - Clear cache if needed: npm start -- --clear
```

### Admin Endpoints Return 404
```
Problem: Backend admin endpoints not working
Solution: Verify admin router is registered
  - Check: backend/api/app.py includes admin router
  - Restart backend: python -m uvicorn backend.api.app:app --reload
```

### Can't Connect to Backend
```
Problem: Frontend can't reach backend
Solution: Check API_BASE_URL in constants/api.ts
  - Should match your backend address (ngrok URL or localhost:8000)
  - Update if needed, restart frontend
```

---

## File Locations for Reference

```
Core Implementation:
├─ backend/api/routes_admin.py ..................... Admin endpoints
├─ backend/services/db_service.py ................. DB query functions
├─ backend/api/app.py ............................. Router registration
├─ visionauth-mobile/app/admin.tsx ................. Admin key screen
├─ visionauth-mobile/app/admin/sessions.tsx ....... Sessions list
├─ visionauth-mobile/app/admin/session/[id].tsx ... Session detail
└─ visionauth-mobile/constants/api.ts ............. API helpers

Documentation:
└─ ADMIN_FEATURE_IMPLEMENTATION.md ................ Full documentation
```

---

## What's Next?

### To Deploy/Use in Production
1. Set `ADMIN_KEY` to a strong random key in your deployment environment
2. Use HTTPS to encrypt the admin key in transit
3. Consider adding rate limiting or IP whitelisting
4. Monitor admin access (add logging to track who accesses what)

### To Extend This Feature
- Add user authentication if you need multiple admins with different permissions
- Add search/filtering to sessions list
- Export sessions to CSV/PDF
- Add graphs/dashboards for verification statistics
- Implement real-time audit log updates with WebSocket

### To Integrate with Your Thesis
- Document this feature in your thesis as a practical admin interface
- Reference it as a data inspection tool for verification quality assurance
- Mention it supports monitoring and debugging during development

---

## Need Help?

Refer to: `ADMIN_FEATURE_IMPLEMENTATION.md` for:
- Detailed API documentation
- Complete test plan with curl examples
- Security considerations
- Frontend screen details
- Troubleshooting guide

---

**Summary:** 
You now have a working admin/audit interface for your thesis project. Start the backend, enter the admin key on the `/admin` screen, and browse your KYC sessions and audit logs. No login system needed—simple, clean, and thesis-appropriate.
