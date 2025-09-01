# Frontend Tech Stack

Based on your solution approach document and the multi-tenant, quality-first requirements, here's the recommended frontend stack.

---

## Core Frontend Stack

**Primary Framework**
- **Next.js 14+ (App Router)** — mentioned in your solution approach  
- **TypeScript** — for type safety in a complex multi-tenant system  
- **React 18+** — with concurrent features

**Styling & UI**
- **Tailwind CSS** — rapid development with a consistent design system  
- **Headless UI / Radix UI** — accessible component primitives  
- **Framer Motion** — smooth animations for review interface

**State Management**
- **Zustand** — lightweight, perfect for tenant-specific state  
- **TanStack Query (React Query)** — server state management, caching  
- **Jotai** — atomic state for complex form states

**Real-time & API**
- **Socket.io Client** — real-time pipeline status updates  
- **Axios / Fetch** — API communication with FastAPI backend  
- **SWR** — data fetching with automatic revalidation

**Specialized Components**
- **Monaco Editor** — inline markdown editing  
- **React DnD** — drag-drop content organization  
- **Recharts** — analytics dashboards  
- **React Hook Form** — optimized form handling

**Mobile & PWA**
- **Next PWA** — progressive web app capabilities  
- **Capacitor** — native mobile app wrapper (future)

---

## Architecture Structure

```
frontend/
├── app/                    # Next.js 14 App Router
│   ├── (dashboard)/        # Dashboard routes group
│   ├── (auth)/             # Authentication routes
│   └── api/                # API routes (middleware)
├── components/
│   ├── ui/                 # Reusable UI components
│   ├── dashboard/          # Dashboard-specific components
│   ├── review/             # 15-min review interface
│   └── tenant/             # Multi-tenant components
├── lib/
│   ├── api/                # API client functions
│   ├── auth/               # Authentication logic
│   └── utils/              # Utility functions
├── stores/                 # State management
└── types/                  # TypeScript definitions
```

---

If you want, I can also:
- Generate a `package.json` and `tsconfig.json` skeleton for this stack.  
- Create a starter Next.js + Tailwind + Zustand repo layout with example components.  
- Export this as a GitHub Gist and attach the link.

