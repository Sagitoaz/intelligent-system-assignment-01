import { useEffect, useState, type ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'

export function Layout({ children }: { children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()

  useEffect(() => setMenuOpen(false), [location.pathname])

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-row">
          <NavLink className="brand" to="/">
            <span className="brand-mark">IS</span>
            <span><strong>Intelligent Systems</strong><small>Assignment 01</small></span>
          </NavLink>
          <button
            className="menu-button"
            type="button"
            aria-expanded={menuOpen}
            aria-controls="primary-navigation"
            aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            onClick={() => setMenuOpen((open) => !open)}
          >
            <span /><span /><span />
          </button>
        </div>
        <nav id="primary-navigation" className={menuOpen ? 'nav-open' : ''} aria-label="Primary navigation">
          <NavLink to="/">Home</NavLink>
          <NavLink to="/diabetes">Diabetes</NavLink>
          <NavLink to="/house-price">House price</NavLink>
          <NavLink to="/diabetes/knowledge-graph">Knowledge graph</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
      </header>
      <main>{children}</main>
      <footer>
        <span>Intelligent System Development · Assignment 01</span>
        <span>Local educational demonstration</span>
      </footer>
    </div>
  )
}
