import { useEffect, useLayoutEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { Link, Navigate, useLocation, useNavigate, useParams } from 'react-router-dom';
import StrokeThumb from '../components/StrokeThumb';
import { useLang } from '../i18n';
import { studioNames } from './StudioPage';
import {
  createDrawing, createProject, deleteDrawing, deleteProject, getProject, isPendingImport, listDrawings, listProjects, renameProject,
  type Drawing, type Project,
} from '../lib/studioProjects';

const fmt = (t: number, locale: string) => new Date(t).toLocaleString(locale, { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

const INTRO_DELAY_MS = 450;
const INTRO_MS = 1100;

function Page({ title, sub, desc, back, center, hero, intro, children }: { title: string; sub?: string; desc?: string; hero?: boolean; back: { to: string; label: string }; center?: boolean; intro?: boolean; children: ReactNode }) {
  const titleRef = useRef<HTMLHeadingElement>(null);
  // Entrada: o titulo nasce grande no meio da tela e sobe ate o lugar dele; so depois o resto aparece.
  useLayoutEffect(() => {
    const el = titleRef.current;
    if (!intro || !el || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const r = el.getBoundingClientRect();
    const scale = Math.min(3, (window.innerWidth * 0.8) / r.width);
    const dx = window.innerWidth / 2 - (r.left + r.width / 2);
    const dy = window.innerHeight / 2 - (r.top + r.height / 2);
    const anim = el.animate(
      [{ transform: `translate(${dx}px, ${dy}px) scale(${scale})` }, { transform: `translate(${dx}px, ${dy}px) scale(${scale})`, offset: INTRO_DELAY_MS / (INTRO_DELAY_MS + INTRO_MS) }, { transform: 'none' }],
      { duration: INTRO_DELAY_MS + INTRO_MS, easing: 'cubic-bezier(.65, 0, .25, 1)', fill: 'backwards' },
    );
    return () => anim.cancel();
  }, [intro]);

  return (
    <section className={`hub-page${center ? ' is-centered' : ''}${hero ? ' is-hero' : ''}${intro ? ' has-intro' : ''}`}>
      <Link className="studio-exit" to={back.to}>← {back.label}</Link>
      <div className="hub-head">
        <h1 className="hub-title" ref={titleRef}>{title}</h1>
        {sub && <p className="hub-sub">{sub}</p>}
        {desc && <p className="hub-desc">{desc}</p>}
      </div>
      {children}
    </section>
  );
}

/** Excluir em dois toques: o primeiro pede confirmação. */
function DeleteButton({ label, onConfirm }: { label: string; onConfirm: () => void }) {
  const [asking, setAsking] = useState(false);
  const { tr } = useLang();
  return (
    <button
      type="button"
      className={`hub-delete${asking ? ' is-asking' : ''}`}
      aria-label={asking ? tr(`Confirmar: excluir ${label}`, `Confirm: delete ${label}`) : tr(`Excluir ${label}`, `Delete ${label}`)}
      onClick={() => (asking ? onConfirm() : setAsking(true))}
      onBlur={() => setAsking(false)}
    >
      {asking ? tr('confirmar?', 'confirm?') : tr('excluir', 'delete')}
    </button>
  );
}

export function NewProject() {
  const navigate = useNavigate();
  const { tr } = useLang();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState(false);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    const project = createProject(name, description, studioNames(tr));
    if (!project) {
      setError(true);
      return;
    }
    const drawing = createDrawing(project.id, studioNames(tr));
    navigate(drawing ? `/app/p/${project.id}/d/${drawing.id}` : `/app/p/${project.id}`);
  };
  return (
    <Page title={tr('Novo projeto', 'New project')} back={{ to: '/app', label: tr('início', 'home') }} center>
      <form className="hub-form" onSubmit={submit}>
        <label htmlFor="project-name">{tr('Nome do projeto', 'Project name')}</label>
        <input id="project-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={60} placeholder={tr('Ex.: Suporte para fone', 'E.g.: Headphone stand')} autoFocus autoComplete="off" />
        <label htmlFor="project-desc">{tr('Descrição', 'Description')} <span>{tr('(opcional)', '(optional)')}</span></label>
        <textarea id="project-desc" value={description} onChange={(e) => setDescription(e.target.value)} maxLength={280} rows={3} placeholder={tr('Do que se trata este projeto?', 'What is this project about?')} />
        <button type="submit" className="hub-primary">{tr('criar projeto', 'create project')}</button>
        {error && <p className="studio-cam-error" role="alert">{tr('Sem espaço no navegador para salvar o projeto.', 'Not enough browser storage to save the project.')}</p>}
      </form>
    </Page>
  );
}

function ProjectCard({ project, onRename, onDelete }: { project: Project; onRename: (name: string) => void; onDelete: () => void }) {
  const { tr, locale } = useLang();
  const drawings = listDrawings(project.id);
  const preview = drawings.find((d) => d.strokes.length);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(project.name);
  const locked = isPendingImport(project.id); // importacao ainda nao gravada: sem renomear nem excluir
  const cancelled = useRef(false);
  const thumb = preview ? <StrokeThumb strokes={preview.strokes} /> : <span className="studio-thumb is-empty" aria-hidden="true" />;
  const meta = <span className="hub-card-meta">{plural(drawings.length, tr('desenho', 'drawing'), tr('desenhos', 'drawings'))} · {fmt(project.updatedAt, locale)}</span>;

  const commit = () => {
    if (cancelled.current) return;
    setEditing(false);
    const next = name.trim() || project.name;
    setName(next);
    if (next !== project.name) onRename(next);
  };

  return (
    <li className="hub-card-wrap">
      {editing ? (
        <div className="hub-card is-editing">
          {thumb}
          <input
            className="hub-card-input"
            value={name}
            maxLength={60}
            aria-label={tr('Nome do projeto', 'Project name')}
            autoFocus
            onFocus={(e) => e.currentTarget.select()}
            onChange={(e) => setName(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur();
              if (e.key === 'Escape') {
                cancelled.current = true;
                setName(project.name);
                setEditing(false);
              }
            }}
          />
          {meta}
        </div>
      ) : (
        <>
          <Link className="hub-card" to={`/app/p/${project.id}`}>
            {thumb}
            <span className="hub-card-name">{project.name}</span>
            {meta}
          </Link>
          {!locked && <button type="button" className="hub-edit" aria-label={tr(`Renomear projeto ${project.name}`, `Rename project ${project.name}`)} onClick={() => { cancelled.current = false; setEditing(true); }}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20l1-4L16.5 4.5a2 2 0 0 1 3 3L8 19l-4 1zM14.5 6.5l3 3" /></svg>
          </button>}
        </>
      )}
      {!locked && <DeleteButton label={tr(`projeto ${project.name}`, `project ${project.name}`)} onConfirm={onDelete} />}
    </li>
  );
}

export function ProjectList() {
  const { tr } = useLang();
  const names = useMemo(() => studioNames(tr), [tr]);
  const [projects, setProjects] = useState(() => listProjects(names));
  const location = useLocation();
  const navigate = useNavigate();
  // A animacao so toca ao chegar pela tela de inicio; voltar de um projeto (ou recarregar) mostra a lista direto.
  const [intro] = useState(() => !!(location.state as { intro?: boolean } | null)?.intro);
  useEffect(() => {
    if (intro) navigate(location.pathname, { replace: true, state: null }); // o botao voltar do navegador nao a repete
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Page title={tr('Projetos', 'Projects')} sub={projects.length ? plural(projects.length, tr('projeto', 'project'), tr('projetos', 'projects')) : undefined} back={{ to: '/app', label: tr('início', 'home') }} intro={intro}>
      <ul className="hub-grid">
        <li><Link className="hub-card is-new" to="/app/new"><span aria-hidden="true">＋</span>{tr('Novo projeto', 'New project')}</Link></li>
        {projects.map((p) => (
          <ProjectCard key={p.id} project={p} onRename={(name) => { renameProject(p.id, name); setProjects(listProjects()); }} onDelete={() => { deleteProject(p.id); setProjects(listProjects()); }} />
        ))}
      </ul>
      {projects.length === 0 && <p className="hub-empty">{tr('Nenhum projeto ainda. Crie o primeiro.', 'No projects yet. Create the first one.')}</p>}
    </Page>
  );
}

function DrawingCard({ drawing, onDelete }: { drawing: Drawing; onDelete?: () => void }) {
  const { tr, locale } = useLang();
  return (
    <li className="hub-card-wrap">
      <Link className="hub-card" to={`/app/p/${drawing.projectId}/d/${drawing.id}`}>
        {drawing.strokes.length ? <StrokeThumb strokes={drawing.strokes} /> : <span className="studio-thumb is-empty" aria-hidden="true" />}
        <span className="hub-card-name">{drawing.name}</span>
        <span className="hub-card-meta">{plural(drawing.strokes.length, tr('traço', 'stroke'), tr('traços', 'strokes'))} · {fmt(drawing.updatedAt, locale)}</span>
      </Link>
      {onDelete && <DeleteButton label={drawing.name} onConfirm={onDelete} />}
    </li>
  );
}

export function ProjectView() {
  const { tr } = useLang();
  const { projectId = '' } = useParams();
  const navigate = useNavigate();
  const names = useMemo(() => studioNames(tr), [tr]);
  const project = getProject(projectId, names);
  const [drawings, setDrawings] = useState(() => listDrawings(projectId));
  const [error, setError] = useState(false);
  // O editor que acabou de fechar grava na limpeza, depois deste primeiro render: le de novo ao montar.
  useEffect(() => setDrawings(listDrawings(projectId)), [projectId]);
  if (!project) return <Navigate to="/app/projects" replace />;
  const pending = isPendingImport(project.id); // importacao ainda nao gravada: so leitura

  const add = () => {
    const drawing = createDrawing(project.id, names);
    if (!drawing) {
      setError(true);
      return;
    }
    navigate(`/app/p/${project.id}/d/${drawing.id}`);
  };

  return (
    <Page title={project.name} sub={plural(drawings.length, tr('desenho', 'drawing'), tr('desenhos', 'drawings'))} desc={project.description} hero back={{ to: '/app/projects', label: tr('projetos', 'projects') }}>
      <ul className="hub-grid">
        {!pending && <li><button type="button" className="hub-card is-new" onClick={add}><span aria-hidden="true">＋</span>{tr('Novo', 'New')}</button></li>}
        {drawings.map((d) => (
          <DrawingCard key={d.id} drawing={d} onDelete={pending ? undefined : () => { deleteDrawing(project.id, d.id); setDrawings(listDrawings(project.id)); }} />
        ))}
      </ul>
      {pending && <p className="hub-notice" role="status">{tr('Importação pendente: sem espaço no navegador para gravar. Apague desenhos para liberar espaço e abra de novo; por enquanto estes só podem ser vistos.', 'Import pending: not enough browser storage to save. Delete some drawings to free up space and reopen; for now these can only be viewed.')}</p>}
      {error && <p className="hub-notice" role="alert">{tr('Sem espaço no navegador para criar o desenho.', 'Not enough browser storage to create the drawing.')}</p>}
    </Page>
  );
}

export function Library() {
  const { tr } = useLang();
  const names = useMemo(() => studioNames(tr), [tr]);
  const load = () => listProjects(names).map((p) => ({ project: p, drawings: listDrawings(p.id) })).filter((g) => g.drawings.length);
  const [groups, setGroups] = useState(load);
  useEffect(() => setGroups(load()), []); // eslint-disable-line react-hooks/exhaustive-deps -- le de novo apos a limpeza do editor
  return (
    <Page title={tr('Biblioteca', 'Library')} sub={tr('todos os desenhos, de todos os projetos', 'all drawings, from all projects')} back={{ to: '/app', label: tr('início', 'home') }}>
      {groups.map(({ project, drawings }) => (
        <div key={project.id} className="hub-group">
          <Link className="hub-group-title" to={`/app/p/${project.id}`}>{project.name}</Link>
          <ul className="hub-grid">
            {drawings.map((d) => <DrawingCard key={d.id} drawing={d} />)}
          </ul>
        </div>
      ))}
      {groups.length === 0 && <p className="hub-empty">{tr('Nenhum desenho ainda. Abra um projeto e clique em Novo.', 'No drawings yet. Open a project and click New.')}</p>}
    </Page>
  );
}
