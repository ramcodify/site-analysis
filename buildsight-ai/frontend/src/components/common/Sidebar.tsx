import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Video, Shield, TrendingUp,
  Users, UserCheck, FileText, Settings,
  HardHat, Activity, Wifi, Upload, BookOpen
} from 'lucide-react';

const navItems = [
  { section: 'Overview', items: [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  ]},
  { section: 'Live Monitoring', items: [
    { to: '/live', icon: Video, label: 'Webcam' },
    { to: '/rtsp', icon: Wifi, label: 'RTSP / CCTV' },
    { to: '/upload', icon: Upload, label: 'Video Upload' },
  ]},
  { section: 'Analytics', items: [
    { to: '/safety', icon: Shield, label: 'Safety Analytics' },
    { to: '/progress', icon: TrendingUp, label: 'Progress Analysis' },
  ]},
  { section: 'Identity & Management', items: [
    { to: '/registered-workers', icon: UserCheck, label: 'Registered Workers' },
    { to: '/workers', icon: Users, label: 'Live Tracking' },
  ]},
  { section: 'Knowledge & Reports', items: [
    { to: '/knowledge', icon: BookOpen, label: 'Safety Knowledge' },
    { to: '/reports', icon: FileText, label: 'Reports' },
  ]},
  { section: 'System', items: [
    { to: '/settings', icon: Settings, label: 'Settings' },
  ]},
];

export function Sidebar() {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <HardHat size={24} style={{ color: '#06b6d4' }} />
          <h1>BuildSight AI</h1>
        </div>
        <p>Construction Intelligence</p>
      </div>
      <nav className="sidebar-nav">
        {navItems.map((section) => (
          <div key={section.section} className="nav-section">
            <div className="nav-section-label">{section.section}</div>
            {section.items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `nav-link${isActive ? ' active' : ''}`
                }
              >
                <item.icon size={18} />
                {item.label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div style={{
        padding: '16px 20px',
        borderTop: '1px solid var(--border-primary)',
        fontSize: 11,
        color: 'var(--text-muted)',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}>
        <Activity size={14} />
        <span>v2.4.0 — BuildSight AI (YOLO11 + SFace)</span>
      </div>
    </aside>
  );
}
