window.HELP_IMPROVE_VIDEOJS = false;

// Auto-resize the embedded qualitative-comparison iframe to avoid internal scrollbars / whitespace.
window.addEventListener('message', (event) => {
  if (event.origin !== window.location.origin) return;
  if (!event.data || event.data.type !== 'vpw_visexpert_demo_height') return;
  const iframe = document.getElementById('qual-demo-iframe');
  if (!iframe) return;
  const height = Number(event.data.height);
  if (!Number.isFinite(height) || height <= 0) return;
  iframe.style.height = `${Math.max(720, Math.min(height, 4000))}px`;
});

$(document).ready(function () {
  // Navbar burger toggle (mobile).
  $('.navbar-burger').click(function () {
    $('.navbar-burger').toggleClass('is-active');
    $('.navbar-menu').toggleClass('is-active');
  });

  const options = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 5000,
  };

  bulmaCarousel.attach('.carousel', options);
  bulmaSlider.attach();
});
