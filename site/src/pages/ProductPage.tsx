import { useEffect, useRef, type CSSProperties, type MouseEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import ProductAsciiArt from '../components/ProductAsciiArt';
import { useLang } from '../i18n';
import { runBlast } from '../lib/blast';

export default function ProductPage() {
  const { tr } = useLang();
  const navigate = useNavigate();
  const stopBlast = useRef<() => void>(() => undefined);
  useEffect(() => () => {
    stopBlast.current();
    document.body.classList.remove('is-falling');
  }, []);

  const enterStudio = (event: MouseEvent) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    if (document.body.classList.contains('is-falling')) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      navigate('/app');
      return;
    }
    document.body.classList.add('is-falling');
    stopBlast.current = runBlast(() => {
      navigate('/app');
      // O estudio ja nasce coberto pelo flash branco; so entao soltamos a explosao.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        stopBlast.current();
        document.body.classList.remove('is-falling');
      }));
    });
  };

  return (
    <main className="product-page" id="page-content">
      <section className="product-content">
        <Link className="product-status product-emerge group" style={{ '--emerge': 300 } as CSSProperties} to="/app" onClick={enterStudio} aria-label={tr('Em construção. Abrir o Draw.', 'Under construction. Open Draw.')}>
          <span className="status-dot" />
          {tr('em construção', 'under construction')}
          <span className="status-arrow" aria-hidden="true">↗</span>
        </Link>
        <figure className="construction-art">
          <ProductAsciiArt />
        </figure>
        <h1 className="product-headline product-emerge" style={{ '--emerge': 900 } as CSSProperties}>
          {tr('Do primeiro traço', 'From the first stroke')}
          <br />
          <em>{tr('à próxima geração.', 'to the next generation.')}</em>
        </h1>
        <p className="product-description product-emerge" style={{ '--emerge': 1500 } as CSSProperties}>
          {tr(
            'Estamos construindo uma ponte entre o papel e as ferramentas que projetam e testam o mundo. A ideia começa à mão; o que ela pode se tornar ainda está sendo desenhado.',
            'We are building a bridge between paper and the tools that design and test the world. The idea starts by hand; what it can become is still being drawn.',
          )}
        </p>

        <Link className="vision-cta product-emerge" style={{ '--emerge': 2100 } as CSSProperties} to="/vision">
          {tr('Leia a visão', 'Read the vision')} <span aria-hidden="true">↗</span>
        </Link>
      </section>
    </main>
  );
}
