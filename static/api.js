(function() {
    window.ClockCaptcha = {
        render: function(container, options) {
            const sitekey = options.sitekey || 'demo';
            const callback = options.callback || function() {};
            
            const iframe = document.createElement('iframe');
            iframe.src = '/widget?sitekey=' + sitekey;
            iframe.style.width = '300px';
            iframe.style.height = '400px';
            iframe.style.border = 'none';
            iframe.style.borderRadius = '8px';
            iframe.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
            
            container.appendChild(iframe);
            
            window.addEventListener('message', function(event) {
                if (event.data.type === 'clock-captcha-success') {
                    callback(event.data.token);
                }
            });
        }
    };
    
    document.addEventListener('DOMContentLoaded', function() {
        const captchaElements = document.querySelectorAll('.clock-captcha');
        captchaElements.forEach(function(element) {
            const sitekey = element.getAttribute('data-sitekey');
            const callback = element.getAttribute('data-callback');
            
            ClockCaptcha.render(element, {
                sitekey: sitekey,
                callback: window[callback] || function() {}
            });
        });
    });
})();
