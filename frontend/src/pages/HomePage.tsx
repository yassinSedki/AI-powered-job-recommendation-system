import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useNavigate } from 'react-router-dom';
import { Briefcase, DollarSign, MapPin, TrendingUp, Users, Award } from 'lucide-react';

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 px-4 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-primary to-accent opacity-10" />
        <div className="container relative z-10">
          <div className="max-w-3xl mx-auto text-center space-y-6">
            <h1 className="text-5xl md:text-6xl font-bold">
              Find Your Perfect{' '}
              <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                Job Match
              </span>
            </h1>
            <p className="text-xl text-muted-foreground">
              AI-powered job recommendations and salary predictions tailored to your location and skills
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Button
                size="lg"
                onClick={() => navigate('/recommend')}
                className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
              >
                <Briefcase className="mr-2 h-5 w-5" />
                Find Jobs Now
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => navigate('/predict-salary')}
              >
                <DollarSign className="mr-2 h-5 w-5" />
                Check Salary
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="container">
          <h2 className="text-3xl font-bold text-center mb-12">
            Why Choose JobHunt?
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="p-6 text-center hover:shadow-xl transition-shadow">
              <div className="bg-gradient-to-r from-primary to-accent w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-4">
                <MapPin className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Location-Based</h3>
              <p className="text-muted-foreground">
                Find jobs near you with our interactive map. Set your preferred location and discover opportunities in your area.
              </p>
            </Card>

            <Card className="p-6 text-center hover:shadow-xl transition-shadow">
              <div className="bg-gradient-to-r from-primary to-accent w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2">AI-Powered Matching</h3>
              <p className="text-muted-foreground">
                Our intelligent algorithm analyzes your skills and experience to recommend the best job matches.
              </p>
            </Card>

            <Card className="p-6 text-center hover:shadow-xl transition-shadow">
              <div className="bg-gradient-to-r from-primary to-accent w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-4">
                <DollarSign className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Salary Insights</h3>
              <p className="text-muted-foreground">
                Get accurate salary predictions based on your qualifications, location, and market trends.
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-4">
        <div className="container">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent mb-2">
                10,000+
              </div>
              <p className="text-muted-foreground">Active Jobs</p>
            </div>
            <div>
              <div className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent mb-2">
                98%
              </div>
              <p className="text-muted-foreground">Match Accuracy</p>
            </div>
            <div>
              <div className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent mb-2">
                5,000+
              </div>
              <p className="text-muted-foreground">Happy Users</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-gradient-to-r from-primary to-accent">
        <div className="container text-center">
          <h2 className="text-4xl font-bold text-primary-foreground mb-4">
            Ready to Find Your Dream Job?
          </h2>
          <p className="text-xl text-primary-foreground/90 mb-8">
            Start your journey today with personalized job recommendations
          </p>
          <Button
            size="lg"
            variant="secondary"
            onClick={() => navigate('/recommend')}
          >
            Get Started
          </Button>
        </div>
      </section>
    </div>
  );
}
