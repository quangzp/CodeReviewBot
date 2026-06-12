import { createContext, useContext, useEffect, useState, ReactNode } from 'react'

interface User {
  login: string
  name: string
  avatar_url?: string
}

interface AuthContextType {
  user: User | null
  authEnabled: boolean
  loading: boolean
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  authEnabled: false,
  loading: true,
  logout: async () => {},
})

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [authEnabled, setAuthEnabled] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('http://localhost:8000/auth/me', { credentials: 'include' })
      .then(r => r.json())
      .then(data => {
        setUser(data.user)
        setAuthEnabled(data.auth_enabled)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const logout = async () => {
    await fetch('http://localhost:8000/auth/logout', {
      method: 'POST',
      credentials: 'include',
    })
    setUser(null)
    window.location.href = '/login'
  }

  return (
    <AuthContext.Provider value={{ user, authEnabled, loading, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
