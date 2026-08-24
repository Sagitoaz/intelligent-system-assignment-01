import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <NavLink className="brand" to="/">
          <span className="brand-mark">IS</span>
          <span><strong>Intelligent Systems</strong><small>Assignment 01</small></span>
        </NavLink>
        <nav aria-label="Primary navigation">
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
