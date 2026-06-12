import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import Dashboard from './pages/Dashboard'
import Chat from './pages/Chat'
import NewReview from './pages/NewReview'
import ReviewDetail from './pages/ReviewDetail'
import Login from './pages/Login'
import Developers from './pages/Developers'
import DeveloperDetail from './pages/DeveloperDetail'
import Projects from './pages/Projects'
import NewProject from './pages/NewProject'
import ProjectDetail from './pages/ProjectDetail'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, authEnabled, loading } = useAuth()

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-gray-400 text-sm animate-pulse">Loading...</div>
      </div>
    )
  }

  // If auth is disabled (no GITHUB_OAUTH_CLIENT_ID set), allow everyone through
  if (!authEnabled) return <>{children}</>

  // Auth enabled but not logged in → go to login
  if (!user) return <Navigate to="/login" replace />

  return <>{children}</>
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<ProtectedRoute><Chat /></ProtectedRoute>} />
      <Route path="/reviews" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
      <Route path="/new" element={<ProtectedRoute><NewReview /></ProtectedRoute>} />
      <Route path="/review/:id" element={<ProtectedRoute><ReviewDetail /></ProtectedRoute>} />
      <Route path="/developers" element={<ProtectedRoute><Developers /></ProtectedRoute>} />
      <Route path="/developer/:login" element={<ProtectedRoute><DeveloperDetail /></ProtectedRoute>} />
      <Route path="/projects" element={<ProtectedRoute><Projects /></ProtectedRoute>} />
      <Route path="/projects/new" element={<ProtectedRoute><NewProject /></ProtectedRoute>} />
      <Route path="/projects/:id" element={<ProtectedRoute><ProjectDetail /></ProtectedRoute>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  )
}
