import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Briefcase } from 'lucide-react';

export default function Header() {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    // Removed backdrop-blur and semi-transparent background to avoid blur effect on scroll
    <header className="sticky top-0 z-50 w-full border-b bg-background">
      <div className="container flex h-16 items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-bold text-xl">
          <div className="bg-gradient-to-r from-primary to-accent p-2 rounded-lg">
            <Briefcase className="w-6 h-6 text-primary-foreground" />
          </div>
          <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            JobHunt
          </span>
        </Link>

        <nav className="flex items-center gap-2">
          <Button
            variant={isActive('/') ? 'default' : 'ghost'}
            asChild
          >
            <Link to="/">Home</Link>
          </Button>
          <Button
            variant={isActive('/recommend') ? 'default' : 'ghost'}
            asChild
          >
            <Link to="/recommend">Find Jobs</Link>
          </Button>
          <Button
            variant={isActive('/predict-salary') ? 'default' : 'ghost'}
            asChild
          >
            <Link to="/predict-salary">Salary Check</Link>
          </Button>
        </nav>
      </div>
    </header>
  );
}
