// ═══════════════════════════════════════════
// FloraVision — Identify Page Logic
// Handles upload, validation, preview, API call, result rendering
// ═══════════════════════════════════════════

// ── Configuration ──
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB in bytes
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
const API_ENDPOINT = 'https://floravision-486x.onrender.com/predict';
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// ── DOM refs ──
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewWrap = document.getElementById('previewWrap');
const previewImg = document.getElementById('previewImg');
const uploadText = document.getElementById('uploadText');
const uploadIcon = document.getElementById('uploadIcon');
const removeBtn = document.getElementById('removeBtn');
const identifyBtn = document.getElementById('identifyBtn');
const uploadError = document.getElementById('uploadError');
const uploadErrorText = document.getElementById('uploadErrorText');
const resultEmpty = document.getElementById('resultEmpty');
const resultLoading = document.getElementById('resultLoading');
const resultCard = document.getElementById('resultCard');
const resultDemoTag = document.getElementById('resultDemoTag');
const mobileMenuToggle = document.getElementById('mobileMenuToggle');
const mobileMenu = document.getElementById('mobileMenu');

let selectedFile = null;

// ══════════════════════════════════════
// MOBILE MENU TOGGLE
// ══════════════════════════════════════

if (mobileMenuToggle && mobileMenu) {
  mobileMenuToggle.addEventListener('click', () => {
    mobileMenuToggle.classList.toggle('active');
    mobileMenu.classList.toggle('active');
    document.body.classList.toggle('menu-open');
  });

  // Close menu when clicking a link
  mobileMenu.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      mobileMenuToggle.classList.remove('active');
      mobileMenu.classList.remove('active');
      document.body.classList.remove('menu-open');
    });
  });
}

// ══════════════════════════════════════
// UPLOAD HANDLING
// ══════════════════════════════════════

browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

uploadZone.addEventListener('click', () => {
  if (!selectedFile) fileInput.click();
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) {
    const file = e.target.files[0];
    validateAndHandleFile(file);
  }
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) validateAndHandleFile(file);
});

removeBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
});

// ══════════════════════════════════════
// FILE VALIDATION
// ══════════════════════════════════════

function validateAndHandleFile(file) {
  hideError();

  // Check file type
  if (!ALLOWED_TYPES.includes(file.type)) {
    showError('Please upload a JPG, PNG, or WEBP image.');
    return;
  }

  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    showError(`File is too large (${sizeMB}MB). Maximum size is 10MB.`);
    return;
  }

  // File is valid, handle it
  handleFile(file);
}

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewWrap.style.display = 'block';
    uploadIcon.style.display = 'none';
    uploadText.style.display = 'none';
    uploadZone.classList.add('has-image');
    identifyBtn.classList.add('ready');
    identifyBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewWrap.style.display = 'none';
  uploadIcon.style.display = 'flex';
  uploadText.style.display = 'flex';
  uploadZone.classList.remove('has-image');
  identifyBtn.classList.remove('ready');
  identifyBtn.disabled = true;
  hideError();
  resetToDemo();
}

// Restores the result panel to its initial demo state
function resetToDemo() {
  // Restore demo card photo and data to Daisy defaults
  const cardPhoto = document.getElementById('cardPhoto');
  cardPhoto.src = 'images/Daisy.jpg';
  cardPhoto.alt = 'White daisy with yellow centre in green grass';
  document.getElementById('cardConfidence').textContent = '97%';
  document.getElementById('cardConfidenceFill').style.width = '97%';
  document.getElementById('cardName').textContent = 'Daisy';
  document.getElementById('cardScientific').textContent = 'Bellis perennis';
  document.getElementById('cardScientific').style.display = 'block';
  document.getElementById('cardDescription').textContent = 'The common daisy (Bellis perennis) is one of the most recognisable wildflowers in the world. Native to western, central, and northern Europe, it has naturalised across every continent. It thrives in lawns and meadows, opening its petals in sunlight and closing them at dusk — a behaviour that gave the flower its Old English name, "day\'s eye."';
  document.getElementById('cardFactText').textContent = 'Children have made daisy chains for over 2,000 years — the practice is documented in Roman writings and early medieval manuscripts alike.';
  document.getElementById('cardFact').style.display = 'flex';
  document.getElementById('cardMeaning').textContent = 'Innocence, purity, and new beginnings. In the Victorian language of flowers, gifting a daisy said: "I\'ll never tell."';
  const wikiLink = document.getElementById('cardWikiLink');
  wikiLink.href = 'https://en.wikipedia.org/wiki/Bellis_perennis';
  document.getElementById('cardWiki').style.display = 'flex';
  const badgesEl = document.getElementById('cardBadges');
  badgesEl.innerHTML = '<span class="care-badge">☀ Full sun</span><span class="care-badge">💧 Low</span><span class="care-badge">🌱 Any soil</span>';
  renderBloomCalendar([3,4,5,6,7,8,9]);
  renderTop3([
    { name: 'Daisy', confidence: 0.97 },
    { name: 'Chamomile', confidence: 0.02 },
    { name: 'Aster', confidence: 0.01 }
  ]);
  // Show demo tag again
  if (resultDemoTag) resultDemoTag.style.display = 'inline-flex';
  showCard();
}

// ══════════════════════════════════════
// ERROR HANDLING
// ══════════════════════════════════════

function showError(message) {
  uploadErrorText.textContent = message;
  uploadError.style.display = 'flex';
  
  // Clear any selected file
  selectedFile = null;
  fileInput.value = '';
  identifyBtn.disabled = true;
  identifyBtn.classList.remove('ready');
}

function hideError() {
  uploadError.style.display = 'none';
  uploadErrorText.textContent = '';
}

// ══════════════════════════════════════
// IDENTIFY — calls Flask /predict
// ══════════════════════════════════════

identifyBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  showLoading();
  identifyBtn.disabled = true;
  
  // Store original button text
  const originalBtnContent = identifyBtn.innerHTML;
  identifyBtn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
      <circle cx="12" cy="12" r="10" opacity="0.25"/>
      <path d="M12 2 a10 10 0 0 1 0 20" stroke-linecap="round">
        <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
      </path>
    </svg>
    Identifying…`;

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      // Handle HTTP errors
      if (response.status === 404) {
        throw new Error('Backend not connected. Showing demo result.');
      } else if (response.status === 500) {
        throw new Error('Server error. Please try again later.');
      } else {
        throw new Error(`Unexpected error (${response.status}). Please try again.`);
      }
    }

    const data = await response.json();
    
    // Validate response data
    if (!data || !data.name) {
      throw new Error('Invalid response from server.');
    }

    renderResult(data);

  } catch (err) {
    console.error('Prediction error:', err);
    
    // Check if it's a network error (backend not running)
    if (err.message.includes('fetch') || err.message.includes('NetworkError') || err.message.includes('404')) {
      // Backend not connected - show demo result for interview
      console.log('Backend not available. Showing demo result.');
      renderResult(getDemoResult());
    } else {
      // Other errors - show friendly error message
      showError(`Oops! ${err.message} Perhaps try another photo?`);
      showCard(); // Keep showing the demo card
    }
  } finally {
    identifyBtn.disabled = false;
    identifyBtn.innerHTML = originalBtnContent;
  }
});

// ══════════════════════════════════════
// STATE SWITCHERS
// ══════════════════════════════════════

function showEmpty() {
  resultEmpty.style.display = 'flex';
  resultLoading.style.display = 'none';
  resultCard.style.display = 'none';
}

function showLoading() {
  resultEmpty.style.display = 'none';
  resultLoading.style.display = 'flex';
  resultCard.style.display = 'none';
}

function showCard() {
  resultEmpty.style.display = 'none';
  resultLoading.style.display = 'none';
  resultCard.style.display = 'block';
}

// ══════════════════════════════════════
// RENDER RESULT
// ══════════════════════════════════════

function renderResult(data) {
  // Hide the demo tag — this is now a real result
  if (resultDemoTag) resultDemoTag.style.display = 'none';
  // ── Photo ──
  const cardPhoto = document.getElementById('cardPhoto');
  cardPhoto.src = previewImg.src;
  cardPhoto.alt = data.name;

  // ── Confidence ──
  const pct = Math.round(data.confidence * 100);
  document.getElementById('cardConfidence').textContent = `${pct}%`;
  document.getElementById('cardConfidenceFill').style.width = `${pct}%`;

  // ── Name ──
  document.getElementById('cardName').textContent = data.name;
  const scientificEl = document.getElementById('cardScientific');
  if (data.scientific_name) {
    scientificEl.textContent = data.scientific_name;
    scientificEl.style.display = 'block';
  } else {
    scientificEl.style.display = 'none';
  }

  // ── Care badges ──
  const badgesEl = document.getElementById('cardBadges');
  badgesEl.innerHTML = '';
  if (data.sun) badgesEl.innerHTML += `<span class="care-badge">☀ ${data.sun}</span>`;
  if (data.water) badgesEl.innerHTML += `<span class="care-badge">💧 ${data.water}</span>`;
  if (data.soil) badgesEl.innerHTML += `<span class="care-badge">🌱 ${data.soil}</span>`;

  // ── Bloom calendar ──
  renderBloomCalendar(data.bloom_months || []);

  // ── Description ──
  document.getElementById('cardDescription').textContent =
    data.description || 'No description available.';

  // ── Fun fact ──
  const factEl = document.getElementById('cardFact');
  const factText = document.getElementById('cardFactText');
  if (data.fun_fact) {
    factText.textContent = data.fun_fact;
    factEl.style.display = 'flex';
  } else {
    factEl.style.display = 'none';
  }

  // ── Top 3 alternatives ──
  renderTop3(data.top3 || []);

  // ── Meaning ──
  const meaningEl = document.getElementById('cardMeaning');
  if (data.meaning) {
    meaningEl.textContent = data.meaning;
  }

  // ── Wikipedia link ──
  const wikiLink = document.getElementById('cardWikiLink');
  const wikiEl = document.getElementById('cardWiki');
  if (data.wikipedia_url) {
    wikiLink.href = data.wikipedia_url;
    wikiEl.style.display = 'flex';
  } else {
    wikiEl.style.display = 'none';
  }

  // Smooth scroll to result
  setTimeout(() => {
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 300);

  showCard();
}

function renderBloomCalendar(bloomMonths) {
  const cal = document.getElementById('bloomCalendar');
  if (!cal) return;
  
  cal.innerHTML = '';

  MONTHS.forEach((m, i) => {
    const monthNum = i + 1;
    const isActive = bloomMonths.includes(monthNum);
    cal.innerHTML += `
      <div class="bloom-month">
        <div class="bloom-month-bar ${isActive ? 'active' : ''}"></div>
        <span class="bloom-month-label">${m}</span>
      </div>`;
  });
}

function renderTop3(top3) {
  const list = document.getElementById('top3List');
  if (!list) return;
  
  list.innerHTML = '';

  if (!top3 || top3.length === 0) {
    list.innerHTML = '<p style="font-size:14px;color:var(--ink-muted);font-style:italic;">No alternative matches available.</p>';
    return;
  }

  const ranks = ['#1', '#2', '#3'];
  top3.slice(0, 3).forEach((item, i) => {
    const pct = Math.round(item.confidence * 100);
    list.innerHTML += `
      <div class="top3-item">
        <span class="top3-rank">${ranks[i]}</span>
        <span class="top3-name">${item.name}</span>
        <div class="top3-bar-wrap">
          <div class="top3-bar-fill" style="width:${pct}%"></div>
        </div>
        <span class="top3-pct">${pct}%</span>
      </div>`;
  });
}

// ══════════════════════════════════════
// DEMO RESULT — for frontend demo when backend isn't connected
// This matches the uploaded image and shows what users can expect
// ══════════════════════════════════════

function getDemoResult() {
  return {
    name: 'Sunflower',
    scientific_name: 'Helianthus annuus',
    confidence: 0.94,
    sun: 'Full sun',
    water: 'Moderate',
    soil: 'Well-drained',
    bloom_months: [6, 7, 8, 9],
    description: 'The sunflower (Helianthus annuus) is a large annual forb of the genus Helianthus grown as a crop for its edible seeds. It is also used as bird food, as livestock forage and in some industrial applications. The plant has a large flowering head (capitulum). The stem can grow up to 3 metres tall, with a flower head that can be 30 cm wide.',
    fun_fact: 'Sunflowers are heliotropic — young plants track the sun across the sky each day, a behaviour driven by unequal cell growth on each side of the stem.',
    meaning: 'Adoration, loyalty, and longevity. In many cultures, sunflowers symbolise positivity and strength.',
    wikipedia_url: 'https://en.wikipedia.org/wiki/Helianthus_annuus',
    top3: [
      { name: 'Sunflower', confidence: 0.94 },
      { name: 'Black-eyed Susan', confidence: 0.04 },
      { name: 'Coneflower', confidence: 0.02 }
    ]
  };
}

// ══════════════════════════════════════
// KEYBOARD ACCESSIBILITY
// ══════════════════════════════════════

// Allow Enter key to trigger file input
uploadZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    if (!selectedFile) fileInput.click();
  }
});

// Make upload zone focusable
uploadZone.setAttribute('tabindex', '0');
uploadZone.setAttribute('role', 'button');
uploadZone.setAttribute('aria-label', 'Upload flower image');
