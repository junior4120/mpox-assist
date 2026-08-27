import { NavLink, Outlet } from "react-router-dom";

const navItems = [
  { to: "/", label: "Dashboard" },
  { to: "/analyse", label: "Nouvelle analyse" },
  { to: "/historique", label: "Historique" },
  { to: "/modele", label: "Modèle IA" },
];

export default function Layout() {
  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-blue-900">MPOX-Assist</h1>
            <p className="text-xs text-gray-500">Plateforme intelligente d'aide au triage</p>
          </div>
          <nav className="flex gap-5 text-sm">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  isActive
                    ? "text-blue-800 font-semibold"
                    : "text-gray-500 hover:text-gray-700"
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}