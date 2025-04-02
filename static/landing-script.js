document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const mobileMenuButton = document.querySelector('.mobile-menu-button');
    const mobileMenu = document.querySelector('.mobile-menu');

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');
            mobileMenuButton.setAttribute(
                'aria-expanded',
                mobileMenuButton.getAttribute('aria-expanded') === 'false' ? 'true' : 'false'
            );
        });
    }

    // Pricing toggle
    const pricingToggle = document.getElementById('pricing-toggle');
    const monthlyPrices = document.querySelectorAll('.monthly-price');
    const annualPrices = document.querySelectorAll('.annual-price');

    if (pricingToggle) {
        pricingToggle.addEventListener('click', function() {
            const isAnnual = pricingToggle.getAttribute('aria-checked') === 'true';
            pricingToggle.setAttribute('aria-checked', isAnnual ? 'false' : 'true');
            
            if (isAnnual) {
                // Switch to monthly
                monthlyPrices.forEach(el => el.classList.remove('hidden'));
                annualPrices.forEach(el => el.classList.add('hidden'));
            } else {
                // Switch to annual
                monthlyPrices.forEach(el => el.classList.add('hidden'));
                annualPrices.forEach(el => el.classList.remove('hidden'));
            }
        });
    }

    // FAQ accordion
    const accordionButtons = document.querySelectorAll('.accordion-header');
    
    accordionButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Toggle the active state of this accordion item
            const content = this.nextElementSibling;
            const icon = this.querySelector('.accordion-button');
            
            content.classList.toggle('open');
            icon.classList.toggle('open');
            
            // Close other accordion items
            accordionButtons.forEach(otherButton => {
                if (otherButton !== button) {
                    const otherContent = otherButton.nextElementSibling;
                    const otherIcon = otherButton.querySelector('.accordion-button');
                    
                    otherContent.classList.remove('open');
                    otherIcon.classList.remove('open');
                }
            });
        });
    });

    // Demo form
    const demoForm = document.getElementById('demo-form');
    const demoResponse = document.getElementById('demo-response');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const audioPlayer = document.getElementById('audio-player');
    
    if (demoForm) {
        demoForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const message = document.getElementById('demo-message').value;
            
            if (!message.trim()) {
                return;
            }
            
            // Show a loading state
            demoResponse.innerText = 'Processing your request...';
            
            // Simulate API call - in production this would call the actual API
            setTimeout(() => {
                const responses = [
                    "I'd be happy to help you with that. Is there anything specific you'd like to know about our voice capabilities?",
                    "Thanks for your interest in CSM-Mimi! Our voice synthesis can handle that request flawlessly.",
                    "Great question! CSM-Mimi's advanced AI can process complex queries like this with natural-sounding speech.",
                    "I understand what you're asking. Our platform is designed to provide human-like responses to queries just like yours."
                ];
                
                // Select a random response
                const randomResponse = responses[Math.floor(Math.random() * responses.length)];
                demoResponse.innerText = randomResponse;
                
                // Simulate audio playback (would be real in production)
                audioPlayerContainer.style.display = 'block';
                // For demo purposes we're not actually playing audio
            }, 1000);
        });
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            
            if (target) {
                window.scrollTo({
                    top: target.offsetTop - 80, // Account for fixed header
                    behavior: 'smooth'
                });
                
                // Close mobile menu if open
                if (!mobileMenu.classList.contains('hidden')) {
                    mobileMenu.classList.add('hidden');
                    mobileMenuButton.setAttribute('aria-expanded', 'false');
                }
            }
        });
    });
});
