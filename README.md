# VisionAuth — AI-Powered eKYC & Biometric Identity Verification System

> Advanced mobile-first electronic Know Your Customer (eKYC) platform using biometric authentication, OCR document analysis, liveness detection, and facial verification.

---

# 📱 Overview

VisionAuth is a full-stack biometric identity verification system designed as a bachelor thesis project focused on secure digital onboarding and electronic identity verification.

The platform simulates a real-world fintech/banking eKYC workflow similar to Revolut, Binance, or digital banking onboarding systems.

Users can:

- Scan a Romanian identity card
- Extract and validate identity data using OCR
- Capture a live selfie
- Complete randomized liveness challenges
- Perform AI-powered face matching
- Receive a final verification decision

Administrators can:

- Access a dedicated web-based admin dashboard
- Review verification sessions
- Inspect biometric evidence
- Analyze audit logs
- Manually approve or reject suspicious sessions

---

# ✨ Main Features

## 🔍 OCR Identity Document Processing

- Romanian ID card detection
- OCR text extraction
- Structured field parsing
- Identity validation
- Automatic field normalization

### Extracted fields

- First name
- Last name
- CNP
- Sex
- Nationality
- Address
- Document series/number
- Validity dates

---

## 🧠 AI Face Matching

The system compares:

- ID document portrait
- Live captured selfie

Using:

- Face embeddings
- Euclidean distance similarity
- Configurable biometric thresholds

### Security logic

- Selfie gate threshold
- Final verification threshold
- Manual review routing
- Multi-stage verification

---

## 🎭 Advanced Randomized Liveness Detection

The platform supports randomized anti-spoofing challenges such as:

- Blink detection
- Smile detection
- Turn head left/right
- Look up/down
- Raise hand
- Show open palm
- Show two fingers
- Touch nose

The challenge is generated dynamically for every session to reduce replay attacks and spoofing attempts.

---

## 🛡️ Security System

### Security Strikes

The system tracks suspicious biometric failures.

Examples:
- Multiple failed selfie attempts
- Invalid face matching
- Multiple faces detected
- Invalid liveness response

After 3 security strikes:
- Session becomes locked
- Verification is rejected
- Audit trail is preserved

---

## 🗂️ Admin Dashboard

Dedicated web admin interface built with React + Vite.

### Features

- Session management
- Search and filtering
- Manual verification
- Approve / Reject actions
- Audit log timeline
- Identity inspection
- Media preview
- Biometric comparison

---

# 🏗️ System Architecture

```text
Mobile App (Expo React Native)
        ↓
FastAPI Backend
        ↓
PostgreSQL Database
        ↓
Admin Web Dashboard (React + Vite)
