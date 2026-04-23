import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import RewriteModal from './components/RewriteModal';
import HomePage from './pages/HomePage';
import MethodologyPage from './pages/MethodologyPage';
import ModelPage from './pages/ModelPage';
import AdminPage from './pages/AdminPage';

export default function App() {
  return (
    <BrowserRouter>
      <div dir="rtl" lang="he" className="flex h-screen flex-col bg-pattern overflow-hidden">
        <Header />
        <main className="flex flex-1 flex-col min-h-0 overflow-y-auto">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/methodology" element={<MethodologyPage />} />
            <Route path="/model" element={<ModelPage />} />
            <Route path="/admin" element={<AdminPage />} />
          </Routes>
        </main>
        <RewriteModal />
      </div>
    </BrowserRouter>
  );
}
