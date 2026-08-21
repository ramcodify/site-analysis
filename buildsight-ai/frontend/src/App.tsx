import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/common/Sidebar';
import Dashboard from './pages/Dashboard';
import LiveMonitoring from './pages/LiveMonitoring';
import RTSPSources from './pages/RTSPSources';
import UploadAnalysis from './pages/UploadAnalysis';
import SafetyAnalytics from './pages/SafetyAnalytics';
import ProgressAnalysis from './pages/ProgressAnalysis';
import RegisteredWorkers from './pages/RegisteredWorkers';
import Workers from './pages/Workers';
import Violations from './pages/Violations';
import DangerZones from './pages/DangerZones';
import SafetyKnowledge from './pages/SafetyKnowledge';
import Reports from './pages/Reports';
import Settings from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/live" element={<LiveMonitoring />} />
            <Route path="/rtsp" element={<RTSPSources />} />
            <Route path="/upload" element={<UploadAnalysis />} />
            <Route path="/safety" element={<SafetyAnalytics />} />
            <Route path="/progress" element={<ProgressAnalysis />} />
            <Route path="/registered-workers" element={<RegisteredWorkers />} />
            <Route path="/workers" element={<Workers />} />
            <Route path="/violations" element={<Violations />} />
            <Route path="/danger-zones" element={<DangerZones />} />
            <Route path="/knowledge" element={<SafetyKnowledge />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
