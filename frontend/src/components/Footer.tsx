import { Link } from "react-router-dom";
import { Briefcase, Github, Linkedin } from "lucide-react";

export default function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t bg-card text-card-foreground">
      <div className="container py-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <div className="bg-gradient-to-r from-primary to-accent p-2 rounded-lg">
              <Briefcase className="w-5 h-5 text-primary-foreground" aria-hidden="true" />
            </div>
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              JobHunt
            </span>
          </Link>

          <div className="flex items-center gap-6">
            <nav className="hidden sm:flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
              <Link to="/" className="hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md transition-colors">Home</Link>
              <Link to="/recommend" className="hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md transition-colors">Find Jobs</Link>
              <Link to="/predict-salary" className="hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md transition-colors">Salary Check</Link>
            </nav>
            
            <div className="flex items-center gap-3">
              <a
                href="https://www.linkedin.com/in/yassin-sedki-a2657b252"
                aria-label="LinkedIn"
                target="_blank"
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md p-2"
              >
                <span className="sr-only">LinkedIn</span>
                <Linkedin className="w-5 h-5" aria-hidden="true" />
              </a>
              <a
                href="https://github.com/yassinSedki"
                aria-label="GitHub"
                target="_blank"
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md p-2"
              >
                <span className="sr-only">GitHub</span>
                <Github className="w-5 h-5" aria-hidden="true" />
              </a>
            </div>
          </div>
        </div>

        <div className="mt-4 text-xs text-muted-foreground">
          © {year} JobHunt. All rights reserved. 
          Yassin Sedki
        </div>
      </div>
    </footer>
  );
}