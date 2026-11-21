import { useEffect, useRef, useState } from 'react';
import {
  animate,
  motion,
  useMotionTemplate,
  useMotionValue,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
  useTransform,
} from 'motion/react';
import { flushSync } from 'react-dom';
import { Link, useLocation } from 'react-router-dom';
import { useLang } from '../i18n';

const MORPH_DISTANCE = 130;
const HEADER_REVEAL = 36;
const LIFT_RANGE = 60;
const HOME_LIFT = 32;
const TEAR_STEPS = 24;
const TEAR_DURATION = 1100;

function tearPolygon(stage: 'start' | 'line' | 'full') {
  const jitter = (index: number, salt: number) => Math.sin(index * 12.9898 + salt * 78.233) * 0.5;
  const side = (direction: 1 | -1, salt: number) => Array.from({ length: TEAR_STEPS + 1 }, (_, index) => {
    const progress = index / TEAR_STEPS;
    const t = stage === 'start' ? 0 : stage === 'line' ? progress : -0.2 + progress * 1.4;
    const offset = stage === 'full' ? 110 : stage === 'start' ? 0 : 0.5 + jitter(index, salt) * 0.8;
    const wobble = direction * offset;
    return `${100 * t + wobble}% ${100 * t - wobble}%`;
  });
  return `polygon(${[...side(1, 1), ...side(-1, 2).reverse()].join(', ')})`;
}

function scrollProgress(value: number) {
  return Math.min(1, Math.max(0, value / MORPH_DISTANCE));
}

function startFontSize() {
  if (window.matchMedia('(max-width: 640px)').matches) {
    return Math.max(92, Math.min(window.innerWidth * 0.24, 148));
  }
  return Math.max(112, Math.min(window.innerWidth * 0.19, 224));
}

function compactFontSize() {
  return window.matchMedia('(max-width: 640px)').matches ? 22 : 26;
}

function readSavedTheme(): boolean {
  try {
    return localStorage.getItem('draw-theme') === 'dark';
  } catch {
    return false;
  }
}

interface SiteHeaderProps {
  onRepeatProduct: () => void;
}

export default function SiteHeader({ onRepeatProduct }: SiteHeaderProps) {
  const { scrollY } = useScroll();
  const reduceMotion = useReducedMotion();
  const location = useLocation();
  const { lang, setLang, tr } = useLang();
  const isHome = location.pathname === '/';
  const isVision = location.pathname === '/vision';
  const isProduct = location.pathname === '/product';
  const isPapers = location.pathname === '/papers';
  const [isScrolled, setIsScrolled] = useState(!isHome);
  const [brandHovered, setBrandHovered] = useState(false);
  const [cardHovered, setCardHovered] = useState(false);
  const [brandFocused, setBrandFocused] = useState(false);
  const [webcamHovered, setWebcamHovered] = useState(false);
  const [webcamCardHovered, setWebcamCardHovered] = useState(false);
  const [webcamFocused, setWebcamFocused] = useState(false);
  const [webcamPinned, setWebcamPinned] = useState(false);
  const [webcamDismissed, setWebcamDismissed] = useState(false);
  const [dark, setDark] = useState(readSavedTheme);
  const [, setViewportRevision] = useState(0);
  const hoverCloseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const webcamAnchorRef = useRef<HTMLDivElement>(null);
  const showWebcamCard = isHome && !isScrolled && (webcamPinned || (!webcamDismissed && (webcamHovered || webcamCardHovered || webcamFocused)));
  const showDefinition = (brandHovered || cardHovered || brandFocused) && !showWebcamCard;

  function cancelHoverClose() {
    if (hoverCloseTimer.current) clearTimeout(hoverCloseTimer.current);
    hoverCloseTimer.current = null;
  }

  function scheduleHoverClose() {
    cancelHoverClose();
    hoverCloseTimer.current = setTimeout(() => {
      setBrandHovered(false);
      setCardHovered(false);
      setWebcamHovered(false);
      setWebcamCardHovered(false);
      setWebcamDismissed(false);
    }, 160);
  }

  const progress = useTransform(scrollY, (value) =>
    isHome ? scrollProgress(value) : 1,
  );
  const headerPeek = useMotionValue(0);
  const headerOpacity = useTransform(() =>
    isHome ? Math.max(headerPeek.get(), Math.min(1, Math.max(0, scrollY.get() / HEADER_REVEAL))) : 1,
  );
  const headerY = useTransform(headerOpacity, (amount) => -10 * (1 - amount));
  const headerPointer = useTransform(headerOpacity, (amount) => (amount < 0.5 ? 'none' : 'auto'));
  const homeLift = useTransform(scrollY, (value) => (isHome ? -HOME_LIFT * (1 - Math.min(1, Math.max(0, value / LIFT_RANGE))) : 0));
  const brandScale = useTransform(progress, (amount) => {
    const ratio = compactFontSize() / startFontSize();
    return 1 + (ratio - 1) * amount;
  });
  const brandBaseY = useTransform(progress, (amount) => {
    const height = startFontSize() * 0.83;
    const scale = 1 + (compactFontSize() / startFontSize() - 1) * amount;
    const centered = -height / 2;
    const compact = 18 + (height * scale) / 2 - height / 2 - window.innerHeight / 2;
    return centered + (compact - centered) * amount;
  });
  const brandY = useTransform(() => brandBaseY.get() + homeLift.get());
  const productOpacity = useTransform(progress, [0.62, 0.94], [0, 1]);
  const productX = useTransform(progress, [0.62, 0.94], [-6, 0]);
  const webcamOpacity = useTransform(scrollY, [0, 40], [1, 0]);
  const inverseBrandScale = useTransform(() => 1 / brandScale.get());
  const glassAmount = useTransform(progress, [0, 0.22], [0, 44]);
  const borderAmount = useTransform(progress, [0, 0.18], [0, 11]);
  const blurAmount = useTransform(() =>
    isHome ? 28 * Math.min(1, progress.get() / 0.2) : 28 * Math.min(1, Math.max(0, scrollY.get() / 24)),
  );
  const headerGutter = useTransform(progress, (amount) =>
    (window.matchMedia('(max-width: 640px)').matches ? 16 : 64) * amount,
  );
  const headerMaxWidth = useTransform(progress, [0, 1], [1440, 1160]);
  const headerWidth = useMotionTemplate`min(calc(100% - ${headerGutter}px), ${headerMaxWidth}px)`;
  const navHeight = useTransform(progress, (amount) => {
    const mobile = window.matchMedia('(max-width: 640px)').matches;
    const initial = mobile ? 52 : 64;
    const compact = mobile ? 46 : 52;
    return initial + (compact - initial) * amount;
  });
  const themeButtonScale = useTransform(progress, (amount) => {
    const mobile = window.matchMedia('(max-width: 640px)').matches;
    const compact = mobile ? 30 / 34 : 34 / 42;
    return 1 + (compact - 1) * amount;
  });
  const navPadding = useTransform(progress, (amount) => {
    const mobile = window.matchMedia('(max-width: 640px)').matches;
    const initial = mobile ? 18 : Math.max(22, Math.min(window.innerWidth * 0.052, 70));
    const compact = mobile ? 12 : Math.max(16, Math.min(window.innerWidth * 0.02, 28));
    return initial + (compact - initial) * amount;
  });
  const navRadius = useTransform(progress, [0, 0.2], [0, 18]);
  const navBackground = useMotionTemplate`color-mix(in srgb, var(--paper) ${glassAmount}%, transparent)`;
  const navBorder = useMotionTemplate`color-mix(in srgb, var(--ink) ${borderAmount}%, transparent)`;
  const navBlur = useTransform(blurAmount, (amount) => (amount < 0.5 ? 'none' : `blur(${amount}px) saturate(138%)`));
  const navShadowAlpha = useTransform(progress, [0, 0.25], [0, 0.045]);
  const navShadow = useMotionTemplate`0 7px 25px rgba(34, 35, 31, ${navShadowAlpha}), inset 0 1px rgba(255, 255, 255, ${navShadowAlpha})`;

  useMotionValueEvent(scrollY, 'change', (value) => {
    setIsScrolled(!isHome || value > 8);
    if (isHome && value <= 0 && headerPeek.get() > 0 && !headerPeek.isAnimating() && !(document.querySelector('.site-header')?.contains(document.activeElement) && document.activeElement?.matches(':focus-visible'))) {
      if (reduceMotion) headerPeek.set(0);
      else animate(headerPeek, 0, { duration: 0.3 });
    }
  });

  useEffect(() => {
    headerPeek.set(0);
    if (!isHome) return;
    let gated = false;
    let idle = 0;
    const showHeader = () => {
      if (reduceMotion) headerPeek.set(1);
      else animate(headerPeek, 1, { duration: 0.5, ease: [0.22, 0.61, 0.36, 1] });
    };
    let cap = 0;
    const peek = (event: Event) => {
      if (!event.cancelable) return false;
      event.preventDefault();
      showHeader();
      gated = true;
      window.clearTimeout(cap);
      cap = window.setTimeout(() => { gated = false; }, 1000);
      return true;
    };
    const atRest = () => window.scrollY <= 0 && headerPeek.get() < 0.5 && !gated;
    const onWheel = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) return;
      if (gated) {
        event.preventDefault();
        window.clearTimeout(idle);
        idle = window.setTimeout(() => { gated = false; }, 140);
        return;
      }
      if (event.deltaY > 0 && atRest()) peek(event);
    };
    let touchY = 0;
    const onTouchStart = (event: TouchEvent) => { touchY = event.touches[0].clientY; };
    const onTouchMove = (event: TouchEvent) => {
      if (gated) { if (event.cancelable) event.preventDefault(); return; }
      if (touchY - event.touches[0].clientY > 12 && atRest()) peek(event);
    };
    const onTouchEnd = () => { window.clearTimeout(idle); idle = window.setTimeout(() => { gated = false; }, 140); };
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (['ArrowDown', 'PageDown', ' ', 'End'].includes(event.key) && atRest()) {
        const target = event.target as HTMLElement | null;
        if (target && /^(INPUT|TEXTAREA|SELECT|BUTTON|A)$/.test(target.tagName)) return;
        if (peek(event)) idle = window.setTimeout(() => { gated = false; }, 300);
      }
    };
    window.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('touchstart', onTouchStart, { passive: true });
    window.addEventListener('touchmove', onTouchMove, { passive: false });
    window.addEventListener('touchend', onTouchEnd, { passive: true });
    window.addEventListener('keydown', onKey);
    return () => {
      window.clearTimeout(idle);
      window.clearTimeout(cap);
      window.removeEventListener('wheel', onWheel);
      window.removeEventListener('touchstart', onTouchStart);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onTouchEnd);
      window.removeEventListener('keydown', onKey);
    };
  }, [isHome, headerPeek, reduceMotion]);

  useEffect(() => {
    setIsScrolled(!isHome);
    setBrandHovered(false);
    setCardHovered(false);
    setBrandFocused(false);
    setWebcamHovered(false);
    setWebcamCardHovered(false);
    setWebcamFocused(false);
    setWebcamPinned(false);
    setWebcamDismissed(false);
  }, [isHome, location.pathname]);

  useEffect(() => {
    if (!isScrolled) return;
    setWebcamHovered(false);
    setWebcamCardHovered(false);
    setWebcamFocused(false);
    setWebcamPinned(false);
    setWebcamDismissed(false);
  }, [isScrolled]);

  useEffect(() => {
    if (!webcamPinned) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (!webcamAnchorRef.current?.contains(event.target as Node)) setWebcamPinned(false);
    };
    document.addEventListener('pointerdown', closeOnOutsidePointer);
    return () => document.removeEventListener('pointerdown', closeOnOutsidePointer);
  }, [webcamPinned]);

  useEffect(() => {
    const updateViewport = () => setViewportRevision((revision) => revision + 1);
    window.addEventListener('resize', updateViewport);
    return () => window.removeEventListener('resize', updateViewport);
  }, []);

  useEffect(() => {
    document.body.classList.toggle('dark', dark);
    document.querySelector('meta[name="theme-color"]')?.setAttribute(
      'content',
      dark ? '#1b1c19' : '#f7f7f4',
    );
    try {
      localStorage.setItem('draw-theme', dark ? 'dark' : 'light');
    } catch {
      // The visual toggle still works when browser storage is unavailable.
    }
  }, [dark]);

  useEffect(() => () => cancelHoverClose(), []);

  useEffect(() => {
    if (!showDefinition && !showWebcamCard) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setBrandHovered(false);
        setBrandFocused(false);
        setWebcamHovered(false);
        setWebcamCardHovered(false);
        setWebcamFocused(false);
        setWebcamPinned(false);
        setWebcamDismissed(false);
      }
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [showDefinition, showWebcamCard]);

  return (
    <>
      <motion.header
        className={`site-header fixed left-1/2 top-2 z-50 ${!isHome || isScrolled ? 'is-scrolled' : ''}`}
        onFocusCapture={() => {
          if (!isHome || headerPeek.get() >= 1) return;
          if (reduceMotion) headerPeek.set(1);
          else animate(headerPeek, 1, { duration: 0.25 });
        }}
        onBlurCapture={(event) => {
          if (!isHome || window.scrollY > 0 || headerPeek.get() <= 0) return;
          if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) return;
          if (reduceMotion) headerPeek.set(0);
          else animate(headerPeek, 0, { duration: 0.3 });
        }}
        style={{ width: headerWidth, maxWidth: headerMaxWidth, x: '-50%', y: headerY, opacity: headerOpacity, pointerEvents: headerPointer }}
      >
        <motion.nav
          className="nav-shell flex w-full items-center justify-between gap-4 border text-ink shadow-[0_7px_25px_rgba(34,35,31,0.04)]"
          style={{
            height: navHeight,
            paddingInline: navPadding,
            borderRadius: navRadius,
            backgroundColor: navBackground,
            borderColor: navBorder,
            backdropFilter: navBlur,
            boxShadow: navShadow,
          }}
          aria-label={tr('Navegação principal', 'Main navigation')}
        >
          <div className="nav-left flex items-center gap-3 sm:gap-6">
            <Link
              className={`nav-link vision-link font-mono text-[12px] tracking-[.025em] sm:text-[16px] text-muted transition-colors duration-300 hover:text-ink${isVision ? ' is-active' : ''}`}
              to="/vision"
              aria-current={isVision ? 'page' : undefined}
            >
              vision
            </Link>
            <Link
              className={`nav-link vision-link font-mono text-[12px] tracking-[.025em] sm:text-[16px] text-muted transition-colors duration-300 hover:text-ink${isPapers ? ' is-active' : ''}`}
              to="/papers"
              aria-current={isPapers ? 'page' : undefined}
            >
              papers
            </Link>
          </div>
          <div className="nav-right flex items-center gap-3 sm:gap-5">
            <span
              className="location font-mono text-[10px] tracking-[.025em] text-muted sm:text-[16px]"
              tabIndex={0}
            >
              <span className="location-main">São Paulo, Brazil</span>
              <span className="location-more">São José dos Campos</span>
            </span>
            <span className="lang-anchor">
            <motion.button
              className="theme-toggle min-h-[34px] min-w-[56px] rounded-full border border-ink/15 bg-paper/20 px-3 font-mono text-[10px] text-ink transition duration-300 hover:-translate-y-px hover:border-ink/40 hover:bg-ink/5 active:translate-y-0 active:scale-[.97] sm:min-h-[42px] sm:min-w-[78px] sm:px-4 sm:text-[13px]"
              style={{ scaleY: themeButtonScale, transformOrigin: 'center' }}
              type="button"
              aria-label={dark ? tr('Ativar tema claro', 'Switch to light theme') : tr('Ativar tema escuro', 'Switch to dark theme')}
              onClick={() => {
                const root = document.documentElement;
                if (reduceMotion || !document.startViewTransition) {
                  setDark((current) => !current);
                  return;
                }
                root.classList.add('theme-tearing');
                const transition = document.startViewTransition(() => {
                  flushSync(() => setDark((current) => !current));
                });
                transition.ready.then(() => {
                  root.animate(
                    {
                      clipPath: [tearPolygon('start'), tearPolygon('line'), tearPolygon('full')],
                      offset: [0, 0.42, 1],
                      easing: ['cubic-bezier(.45,.05,.3,1)', 'cubic-bezier(.65,0,.2,1)'],
                    },
                    { duration: TEAR_DURATION, fill: 'forwards', pseudoElement: '::view-transition-new(root)' },
                  );
                });
                transition.finished.catch(() => undefined).finally(() => root.classList.remove('theme-tearing'));
              }}
            >
              {dark ? 'light' : 'dark'}
            </motion.button>
            <span className="lang-menu" role="group" aria-label={tr('Idioma', 'Language')}>
              <button type="button" aria-pressed={lang === 'pt'} onClick={() => setLang('pt')}>PT</button>
              <button type="button" aria-pressed={lang === 'en'} onClick={() => setLang('en')}>EN</button>
            </span>
          </span>
          </div>
        </motion.nav>
      </motion.header>

      <motion.div
        className={`brand-lockup${!isHome || isScrolled ? ' is-scrolled' : ''}`}
        style={{ top: '50%', y: brandY, x: '-50%', scale: brandScale }}
      >
        <motion.div
          className="wordmark-trigger"
          onHoverStart={() => {
            cancelHoverClose();
            setBrandHovered(true);
          }}
          onHoverEnd={scheduleHoverClose}
          onFocus={() => {
            cancelHoverClose();
            setBrandFocused(true);
          }}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
              setBrandFocused(false);
            }
          }}
        >
          <Link
            className="wordmark relative z-[1] block font-sans font-extrabold leading-[.83] tracking-[-.035em] text-ink no-underline"
            to="/"
            aria-label={tr('Draw, início', 'Draw, home')}
          >
            {Array.from('Draw').map((letter, index) => (
              <span className="wordmark-letter" aria-hidden="true" key={index}>{letter}</span>
            ))}
          </Link>
          {isHome && (
            <div className="webcam-interaction" ref={webcamAnchorRef}>
              <motion.div
                className="webcam-anchor"
                aria-hidden={isScrolled}
                style={{
                  opacity: reduceMotion ? (isScrolled ? 0 : 1) : webcamOpacity,
                  pointerEvents: isScrolled ? 'none' : 'auto',
                }}
                onHoverStart={() => {
                  cancelHoverClose();
                  setBrandHovered(false);
                  setCardHovered(false);
                  setBrandFocused(false);
                  setWebcamHovered(true);
                }}
                onHoverEnd={scheduleHoverClose}
              >
                <motion.button
                  className="webcam-trigger"
                  type="button"
                  aria-label={tr('Sobre a webcam do Draw', 'About the Draw webcam')}
                  aria-controls="webcam-info-card"
                  aria-expanded={showWebcamCard}
                  tabIndex={isScrolled ? -1 : 0}
                  title={tr('Sobre a webcam do Draw', 'About the Draw webcam')}
                  onFocus={() => {
                    cancelHoverClose();
                    setBrandFocused(false);
                    setWebcamFocused(true);
                  }}
                  onBlur={() => {
                    setWebcamFocused(false);
                    setWebcamDismissed(false);
                  }}
                  onClick={() => {
                    if (webcamPinned) setWebcamDismissed(true);
                    setWebcamPinned(!webcamPinned);
                  }}
                  whileHover={{ y: -1 }}
                  transition={{ duration: 0.18, ease: 'easeOut' }}
                >
                  <svg className="webcam-icon" viewBox="0 0 64 40" aria-hidden="true">
                    <rect x="5" y="4" width="54" height="26" rx="9" />
                    <circle className="webcam-lens" cx="32" cy="17" r="8" />
                    <circle className="webcam-aperture" cx="32" cy="17" r="3" />
                    <circle className="webcam-status" cx="51" cy="12" r="2.2" />
                    <circle className="webcam-rec" cx="51" cy="12" r="1.1" />
                    <path d="M27 30v3c0 1.8 1.4 3 3 3h4c1.7 0 3-1.3 3-3v-3" />
                  </svg>
                </motion.button>
              </motion.div>
              <motion.aside
                id="webcam-info-card"
                className="webcam-card-anchor definition-card"
                aria-label={tr('Sobre a webcam do Draw', 'About the Draw webcam')}
                aria-hidden={!showWebcamCard}
                initial={false}
                animate={showWebcamCard ? 'visible' : 'hidden'}
                style={{ scale: inverseBrandScale }}
                onHoverStart={() => {
                  cancelHoverClose();
                  setBrandHovered(false);
                  setCardHovered(false);
                  setBrandFocused(false);
                  setWebcamCardHovered(true);
                }}
                onHoverEnd={scheduleHoverClose}
                variants={{
                  hidden: { opacity: 0, y: 8, pointerEvents: 'none' },
                  visible: { opacity: 1, y: 0, pointerEvents: 'auto' },
                }}
                transition={{ type: 'spring', stiffness: 320, damping: 28 }}
              >
                <h1>webcam</h1>
                <p className="definition-kind"><em>{tr('instrumento', 'instrument')}</em> · {tr('captura', 'capture')}</p>
                <p className="definition-phrase">
                  {tr('Acompanha o papel e transforma cada traço em tinta digital, em tempo real.', 'Follows the paper and turns every stroke into digital ink, in real time.')}
                </p>
                <p className="definition-note">{tr('A primeira camada do Draw.', 'The first layer of Draw.')}</p>
              </motion.aside>
            </div>
          )}
        </motion.div>
        <motion.div
          className={`product-label font-mono text-muted${isProduct ? ' is-active' : ''}`}
          style={{
            opacity: productOpacity,
            x: productX,
            y: '-50%',
            scale: inverseBrandScale,
            transformOrigin: 'left center',
          }}
          onHoverStart={() => {
            cancelHoverClose();
            setBrandHovered(false);
            setCardHovered(false);
          }}
          whileHover={{ filter: 'brightness(1.25)' }}
          transition={{ type: 'spring', stiffness: 380, damping: 24 }}
        >
          <Link
            to="/product"
            aria-current={isProduct ? 'page' : undefined}
            onClick={(event) => {
              if (isProduct) {
                event.preventDefault();
                onRepeatProduct();
              }
            }}
          >
            <span aria-hidden="true">·</span> <span className="product-word">product</span>
          </Link>
        </motion.div>
        <motion.div className="definition-card-anchor" style={{ x: '-50%', scale: inverseBrandScale }}>
          <motion.aside
            className="definition-card border border-white/70 bg-paper/65 p-5 text-ink shadow-[0_18px_54px_rgba(36,37,32,.1)] backdrop-blur-2xl"
            aria-label={tr('Definição de Draw', 'Definition of Draw')}
            aria-hidden={!showDefinition}
            initial={false}
            animate={showDefinition ? 'visible' : 'hidden'}
            onHoverStart={() => {
              cancelHoverClose();
              setCardHovered(true);
            }}
            onHoverEnd={scheduleHoverClose}
            variants={{
              hidden: { opacity: 0, y: 9, pointerEvents: 'none', transitionEnd: { visibility: 'hidden' } },
              visible: { opacity: 1, y: 0, pointerEvents: 'auto', visibility: 'visible' },
            }}
            transition={{ type: 'spring', stiffness: 320, damping: 28 }}
          >
            <h1>draw</h1>
            <p className="definition-kind"><em>{tr('substantivo', 'noun')}</em> · {tr('conceito', 'concept')}</p>
            <p className="definition-phrase">
              {tr('Traçar uma ideia no papel e transformá-la em algo que pode ser compreendido, testado e levado ao mundo.', 'To trace an idea on paper and turn it into something that can be understood, tested and taken into the world.')}
            </p>
            <p className="definition-note">{tr('Do papel à vida.', 'From paper to life.')}</p>
          </motion.aside>
        </motion.div>
      </motion.div>
    </>
  );
}
