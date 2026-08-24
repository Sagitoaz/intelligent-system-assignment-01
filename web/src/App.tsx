import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { AboutPage } from './pages/AboutPage'
import { DiabetesPage } from './pages/DiabetesPage'
import { HomePage } from './pages/HomePage'
import { HousePricePage } from './pages/HousePricePage'
import { KnowledgeGraphPage } from './pages/KnowledgeGraphPage'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/diabetes" element={<DiabetesPage />} />
        <Route path="/house-price" element={<HousePricePage />} />
        <Route path="/diabetes/knowledge-graph" element={<KnowledgeGraphPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Layout>
  )
}
