"""Comprehensive Pre-Commit Audit Test Suite for the polyglot repo scout (PolyglotRepoIndexer).

Covers:
 1. Python Extraction Gate (imports, classes, signatures, constants / aliases, call edges, docs, fallback)
 2. PHP 8.3 Extraction Gate (namespaces, use, class-likes, methods, PDO SQL, routes, call edges)
 3. JavaScript / TypeScript Extraction Gate (imports, exports, classes, functions, TS types, React, handlers, API calls)
 4. SQL Extraction Gate (DDL, dialect tolerance, migrations)
 5. Walker Gate (ignore rules at depth, symlinks, binary / oversize / minified, encodings, POSIX paths, determinism)
 6. Data Model Gate (serialization round-trip, digest, query helpers, clipping / no file bodies)
 7. Incremental Reuse Gate (content-hash reuse, re-resolution)
 8. Cross-File Resolution Gate (Python, PHP + PSR-4, JS / TS, SQL, embedded SQL, API calls -> routes)
 9. Downstream Hooks Gate (render_outline and to_summary honour their token budgets)
10. Error & Configuration Gate (exception hierarchy, bad files never raise, aindex, stdlib-only imports)
11. Pathological Input Gate (near-linear time on adversarial input)
12. Optional tree-sitter Gate (runs only where tree_sitter is importable)
13. Review Round 1 Regressions (one focused test per confirmed review finding)
14. Review Round 2 Regressions (one focused test per confirmed review finding)
15. Review Round 3 Regressions (one focused test per residual gap)

Stress inputs are scaled down by default; set SCOUT_STRESS_FULL=1 to run them at full size.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

import backend.scout_and_ast as scout
from backend.scout_and_ast import (
    CHARS_PER_TOKEN,
    DEFAULT_IGNORE_DIRS,
    LANGUAGE_BY_EXTENSION,
    MAX_CAPTURED_CHARS,
    MAX_DOC_CHARS,
    MAX_SIGNATURE_CHARS,
    MAX_SQL_PREVIEW_CHARS,
    SCHEMA_VERSION,
    Edge,
    FileRecord,
    IndexerOptions,
    IndexFormatError,
    PolyglotRepoIndexer,
    RepoIndex,
    ScoutConfigurationError,
    ScoutError,
    Symbol,
    estimate_tokens,
    language_for_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLASH_PROMPT_TOKEN_CAP = 90_000  # the scout swarm's hard prompt cap for gemini-3-8-flash


# ======================================================================================
# Helpers
# ======================================================================================


def _write(root: Path, files: Mapping[str, str | bytes]) -> Path:
    """Materialize a synthetic repo; text is written as UTF-8 with LF line endings."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return root


def _indexer(root: Path, **options: Any) -> PolyglotRepoIndexer:
    options.setdefault("use_tree_sitter", False)
    return PolyglotRepoIndexer(root, **options)


def _index(root: Path, **options: Any) -> RepoIndex:
    return _indexer(root, **options).index()


def _record(root: Path, rel: str, text: str | bytes, **options: Any) -> FileRecord:
    _write(root, {rel: text})
    return _indexer(root, **options).index_file(rel)


def _sym(rec: FileRecord, qualified_name: str, kind: str | None = None) -> Symbol:
    found = [
        s for s in rec.symbols
        if s.qualified_name == qualified_name and (kind is None or s.kind == kind)
    ]
    assert found, f"no symbol {qualified_name!r} (kind={kind}) in {rec.path}: " + ", ".join(
        f"{s.kind}:{s.qualified_name}" for s in rec.symbols
    )
    return found[0]


def _names(rec: FileRecord, kind: str) -> list[str]:
    return [s.qualified_name for s in rec.symbols if s.kind == kind]


def _edges(rec: FileRecord, kind: str, source: str | None = None) -> list[Edge]:
    return [e for e in rec.edges if e.kind == kind and (source is None or e.source == source)]


def _targets(rec: FileRecord, kind: str, source: str | None = None) -> set[str]:
    return {e.target for e in _edges(rec, kind, source)}


def _edge(rec: FileRecord, kind: str, target: str) -> Edge:
    found = [e for e in rec.edges if e.kind == kind and e.target == target]
    assert found, f"no {kind} edge to {target!r} in {rec.path}: " + ", ".join(
        sorted(e.target for e in rec.edges if e.kind == kind)
    )
    return found[0]


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _summary_tokens(summary: Mapping[str, Any]) -> int:
    return estimate_tokens(json.dumps(summary))


def _link_dir(link: Path, target: Path) -> None:
    """Create a directory symlink (or an NTFS junction); skip the test when not permitted."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        done = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False
        )
        if done.returncode == 0:
            return
    pytest.skip("directory symlinks / junctions cannot be created on this machine")


# ======================================================================================
# Fixtures (synthetic sources)
# ======================================================================================

PY_SERVICE = r'''"""Module doc."""
from __future__ import annotations
import os, json as j
import a.b.c
from . import sibling
from .models import User, Order as O
from ..core.db import get_conn
from typing import Any, Literal, TypeVar, Callable, TypeAlias
from fastapi import APIRouter

T = TypeVar("T")
Effort = Literal["low", "high"]
Callback: TypeAlias = Callable[[int], None]
type Pair[K, V] = tuple[K, V]
MAX_SIZE = 10_000
TIMEOUT: float = 3.5
logger = None
__all__ = ["Service", "handler"]
router = APIRouter(prefix="/api/v1")
QUERY = "SELECT id FROM accounts WHERE id = %s"

class Color(enum.Enum):
    RED = 1
    GREEN = 2

@dataclass(frozen=True, slots=True)
class Service(Base, Generic[T], metaclass=Meta):
    """Service doc line.

    More detail that must never reach the index.
    """
    name: str
    retries: int = 3
    LIMIT = 5
    objects = Manager()

    class Inner:
        def deep(self): return 1

    def __init__(self, a, /, b: int = 5, *args: Any, key: str = "a-very-long-default-value-that-is-elided", **kw: Any) -> None:
        self.a = a

    @property
    def size(self) -> int:
        return len(self.name)

    @staticmethod
    async def fetch(conn, uid: int, *, timeout: float | None = None) -> dict[str, Any]:
        """Fetch one."""
        cur = conn.cursor()
        cur.execute("SELECT * FROM users u JOIN orders o ON o.uid = u.id WHERE u.id = %s", (uid,))
        cur.execute(QUERY)
        sql = f"DELETE FROM {TABLE} WHERE id = {uid}"
        await conn.execute(sql)
        def nested(x): return helper(x) + json.dumps(x)
        return await asyncio.gather(nested(1), self.size)

@router.get("/users/{user_id}")
async def handler(user_id: int):
    return Service.fetch(None, user_id)

@app.route("/legacy", methods=["GET", "POST"])
def legacy(): pass

urlpatterns = [path("items/<int:pk>/", views.ItemView.as_view()), path("x/", include("x.urls"))]
'''

PY_BROKEN = (
    "import os\n"
    "from .x import (a, b as c)\n"
    "class A(B):\n"
    "    def m(self, x,\n"
    "          y=1) -> int:\n"
    "        return (\n"
    "    async def n(self): pass\n"
    "def top(: pass\n"
)

PHP_CONTROLLER = r'''<?php
declare(strict_types=1);

namespace App\Http\Controllers;

use App\Models\User;
use App\Repositories\{UserRepository, OrderRepository as Orders};
use function App\Support\helper as h;
use const App\Support\VERSION;
use Symfony\Component\Routing\Annotation\Route;

/**
 * Handles users.
 * @package x
 */
#[Route('/users', name: 'users_')]
final class UserController extends BaseController implements Jsonable, \Countable
{
    use LogsActivity, Cacheable { Cacheable::flush as protected flushCache; }

    public const string TABLE = 'users';
    private const A = 1, B = [1, 2];
    protected static ?UserRepository $repo = null;
    public readonly int|string $id;

    public function __construct(
        private readonly \PDO $pdo,
        protected ?Orders $orders = null,
        int $plain = 5,
    ) {}

    /** Show one user. */
    #[Route('/{id}', name: 'show', methods: ['GET', 'HEAD'])]
    public function show(int $id, string ...$rest): ?User
    {
        $stmt = $this->pdo->prepare('SELECT u.*, o.total FROM users u JOIN orders o ON o.user_id = u.id WHERE u.id = ?');
        $stmt->execute([$id]);
        $sql = "UPDATE `audit_log` SET seen = 1 "
             . "WHERE user_id = " . $id;
        $this->pdo->exec($sql);
        $rows = $this->pdo->query(<<<SQL
            SELECT *
              FROM app.sessions
             WHERE user_id = {$id}
            SQL);
        $u = new User($rows);
        $x = User::find($id);
        h($x); strlen("a // not comment"); self::helper(); parent::boot();
        $this->log('x');
        return $u->refresh();
    }

    public function purge(): void
    {
        $this->pdo->exec(<<<'SQL'
            DELETE FROM sessions WHERE expired = 1
            SQL);
        $this->pdo->query("SELECT name FROM `roles` WHERE id = 1");
        // $this->pdo->exec('DROP TABLE commented_out');
        $note = 'class Ghost { function ghost() {} }';
    }

    abstract protected static function &make(array &$data, (A&B)|null $dnf = null): static;

    final public static function helper(): void {}
}

interface Jsonable extends \JsonSerializable, Arrayable { public function toJson(int $flags = 0): string; }

trait LogsActivity { public function log(string $m): void { error_log($m); } }

enum Suit: string implements HasLabel {
    case Hearts = 'H';
    case Spades = 'S';
    const Wild = self::Spades;
    public function label(): string { return match($this) { self::Hearts => 'h', default => 's' }; }
}

function top_level(int $a, $b = 'x)y'): int { return other_fn($a); }
define('APP_ROOT', __DIR__);
const GLOBAL_C = 3;
?>
<html><body class="x">class Fake { function nope() {} }</body></html>
<?php
Route::get('/health', fn () => 'ok');
'''

PHP_NAMESPACES = r'''<?php
namespace Alpha {
    const VERSION = '1';
    function alpha_fn(): void {}
    readonly class Point { public function __construct(public int $x = 0) {} }
}
namespace Beta\Deep {
    abstract class Shape {
        abstract public function area(): float;
        final public function name(): string { return static::class; }
    }
    enum Status { case Active; case Archived; }
}
'''

PHP_ROUTES = r'''<?php
use App\Http\UserController;

Route::prefix('api')->group(function () {
    Route::get('/users', [UserController::class, 'index'])->name('users.index');
    Route::get('/users/{id}', [UserController::class, 'show']);
    Route::post('/users', 'App\Http\UserController@store');
    Route::match(['put', 'patch'], '/users/{id}', [UserController::class, 'store']);
    Route::delete('/users/{id}', [UserController::class, 'destroy']);
    Route::apiResource('orders', OrderController::class);
    Route::resource('photos', PhotoController::class);
});
$app->get('/slim/ping', \App\Http\UserController::class . ':index');
$app->post('/slim/items', function ($request, $response) { return $response; });
'''

TSX_APP = r'''#!/usr/bin/env node
import React, { useState, useEffect as useFx } from 'react';
import * as api from './api/client';
import type { User, Role } from "../types";
import DefaultThing from './thing.js';
import './styles.css';
const legacy = require('legacy-lib');
const { a, b: c } = require("./util");
const lazyMod = () => import('./lazy/Mod');
export { helper, other as renamed } from './helpers';
export * from './all';
export * as ns from './ns';

const API_BASE = '/api/v1';
const client = axios.create({ baseURL: '/api/v2', timeout: 10 });
export const MAX_ITEMS: number = 50;
let counter = 0;

/** A user card. */
export interface CardProps extends BaseProps<User>, Clickable {
  user: User;
  onSelect?: (id: string) => void;
  render(): JSX.Element;
}

export type Id = string | number;
type Handler<T> = (value: T) => void;

export enum Color { Red, Green = 'green', Blue = 4 }
declare namespace Legacy.Stuff { function old(x: number): void; }

export abstract class Repo<T> extends Base<T> implements Readable, Writable<T> {
  static instances = 0;
  private readonly cache: Map<string, T> = new Map();
  #secret = 1;
  constructor(private http: HttpClient, name: string) { super(name); }
  get size(): number { return this.cache.size; }
  set size(v: number) { }
  static create<T>(name: string): Repo<T> { return new ConcreteRepo(name); }
  async *stream(): AsyncGenerator<T> { yield* this.items(); }
  abstract find(id: Id): Promise<T | undefined>;
  handleClick = async (e: MouseEvent): Promise<void> => {
    const res = await fetch(`${API_BASE}/users/${e.detail}/items?x=1`, { method: 'DELETE', headers: { 'X': 'y' } });
    await this.http.post<User>('/api/users', { name: 'x' });
  };
  protected load(id: Id): void {
    const re = /ab+c\/[/]"'/gi; const half = 4 / 2 / 1;
    axios.get('/api/items/' + id + '/detail');
    axios({ url: '/api/raw', method: 'put' });
    client.patch(`/things/${id}`);
    $.ajax({ url: "/api/legacy", type: "POST" });
    $.getJSON('/api/json');
    $.post('/api/form', payload);
    const xhr = new XMLHttpRequest(); xhr.open("GET", "/api/xhr/" + id);
    window.fetch(url, opts);
  }
}

export default function App({ items }: { items: string[] }) {
  const [n, setN] = useState(0);
  useFx(() => { document.title = `n=${n}`; }, [n]);
  const ref = React.useRef(null);
  const handleSave = () => { save(n); };
  document.getElementById('x').addEventListener('click', handleSave);
  window.onresize = function () { relayout(); };
  btn.onclick = handleSave;
  return (
    <div className="app" onClick={handleSave} onMouseEnter={() => setN(n + 1)}>
      <p>Don't panic {n}</p>
      <Card user={u} onSelect={this.props.onPick} onChange={(e) => { track(e); }} />
    </div>
  );
}

export const Card = React.memo(function Card({ user }: CardProps) { return <span>{user.name}</span>; });
export const Row: React.FC<CardProps> = ({ user }) => <li>{user.name}</li>;
export function useThing(id: string) { const [v] = useState(id); return v; }
export async function* gen<T extends object = {}>(a: T, b = 2): AsyncGenerator<T> { yield a; }
function plain(a, b) { return helperFn(a) + b; }
const square = x => x * x;
var legacyFn = function (q) { return q; };
module.exports = { plain, square };
exports.extra = function extra(q) { return q; };
Widget.prototype.draw = function (ctx) { ctx.fill(); };
export = App;
'''

SQL_SCHEMA = r'''-- schema; with a semicolon in a comment
/* block ; comment /* nested */ still */
CREATE TABLE IF NOT EXISTS `users` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `email` VARCHAR(255) NOT NULL DEFAULT 'a;b)c',
  `role` ENUM('admin','user') NOT NULL,
  `key` VARCHAR(50),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_email` (`email`),
  KEY idx_role (role)
) ENGINE=InnoDB;

CREATE TABLE public."orders" (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total NUMERIC(10, 2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    CONSTRAINT fk_ship FOREIGN KEY (ship_country, ship_zip) REFERENCES geo.zones (country, zip) ON DELETE SET NULL ON UPDATE NO ACTION
);

CREATE TABLE [dbo].[Audit Log] ([Id] INT IDENTITY(1,1) PRIMARY KEY, [UserId] INT FOREIGN KEY REFERENCES [dbo].[Users]([Id]));

ALTER TABLE orders ADD COLUMN note TEXT, ADD CONSTRAINT fk_orders_user FOREIGN KEY (user_id) REFERENCES users (id) ON UPDATE CASCADE;
ALTER TABLE `orders` ADD FOREIGN KEY (coupon_id) REFERENCES coupons(id);
ALTER TABLE orders ADD INDEX idx_total (total), ADD UNIQUE INDEX uq_note (note), DROP COLUMN legacy;
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_user ON public.orders USING btree (user_id, lower(note)) WHERE total > 0;
CREATE INDEX ON orders (created_at);
CREATE INDEX idx_my USING HASH ON users (email);
CREATE OR REPLACE VIEW active_users AS SELECT u.id FROM users u JOIN orders o ON o.user_id = u.id WHERE u.role = 'x;y';
DROP TABLE IF EXISTS legacy_a, legacy_b CASCADE;
CREATE FUNCTION touch() RETURNS trigger AS $body$
BEGIN
  UPDATE ghost SET x = 1; -- not a statement
  RETURN NEW;
END;
$body$ LANGUAGE plpgsql;
CREATE TRIGGER trg_touch BEFORE UPDATE ON orders FOR EACH ROW EXECUTE FUNCTION touch();
DELIMITER $$
CREATE PROCEDURE cleanup()
BEGIN
  DELETE FROM sessions WHERE expired = 1;
  CREATE TABLE never_seen (id INT);
END$$
DELIMITER ;
INSERT INTO users (email) VALUES ('x''y;z');
'''

SQL_PATH = "db/migrations/2024_01_15_120000_create_core.sql"

POLYREPO: dict[str, str] = {
    "composer.json": '{"autoload": {"psr-4": {"App\\\\": "src/", "Lib\\\\": ["lib/"]}}}',
    "src/Models/User.php": r'''<?php
namespace App\Models;

use App\Contracts\HasName;
use App\Support\Timestamps;

/** A user row. */
class User extends Model implements HasName
{
    use Timestamps;
    public const TABLE = 'users';
    public function __construct(public readonly int $id, protected string $name = '') {}
    public function name(): string { return $this->name; }
    public static function find(\PDO $pdo, int $id): ?self
    {
        $stmt = $pdo->prepare("SELECT id, name FROM users WHERE id = :id");
        $stmt->execute(['id' => $id]);
        return null;
    }
}
''',
    "src/Models/Model.php": (
        "<?php\nnamespace App\\Models;\n"
        "abstract class Model { abstract public function name(): string; }\n"
    ),
    "src/Contracts/HasName.php": (
        "<?php\nnamespace App\\Contracts;\ninterface HasName { public function name(): string; }\n"
    ),
    "src/Support/Timestamps.php": (
        "<?php\nnamespace App\\Support;\ntrait Timestamps { public function touch(): void {} }\n"
    ),
    "src/Support/helpers.php": (
        "<?php\nnamespace App\\Support;\n"
        "function slugify(string $s): string { return strtolower($s); }\n"
    ),
    "lib/Mailer.php": (
        "<?php\nnamespace Lib;\n"
        "final class Mailer { public function send(string $to): bool { return true; } }\n"
    ),
    "src/Http/UserController.php": r'''<?php
namespace App\Http;

use App\Models\User;
use Lib\Mailer;
use App\Jobs\SendWelcome;
use function App\Support\slugify;

class UserController
{
    public function __construct(private \PDO $pdo, private Mailer $mailer) {}
    public function index(): array { return $this->pdo->query('SELECT * FROM users ORDER BY id')->fetchAll(); }
    public function show(int $id): ?User { $u = User::find($this->pdo, $id); slugify('x'); return $u; }
    public function store(): User {
        $this->pdo->exec("INSERT INTO orders (user_id, total) VALUES (1, 2)");
        $this->pdo->exec("DELETE FROM ghosts WHERE id = 1");
        $m = new Mailer();
        $m->send('a@b.c');
        $job = new SendWelcome();
        return new User(1);
    }
    public function destroy(int $id): void {}
}
''',
    "routes/api.php": PHP_ROUTES,
    "db/schema.sql": (
        "CREATE TABLE users (\n  id INT PRIMARY KEY,\n  name VARCHAR(100) NOT NULL\n);\n"
        "CREATE TABLE orders (\n  id INT PRIMARY KEY,\n"
        "  user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,\n"
        "  total DECIMAL(10,2)\n);\n"
    ),
    "db/migrations/002_add_coupons.sql": (
        "CREATE TABLE coupons (id INT PRIMARY KEY, code VARCHAR(20));\n"
        "ALTER TABLE orders ADD COLUMN coupon_id INT, ADD CONSTRAINT fk_coupon "
        "FOREIGN KEY (coupon_id) REFERENCES coupons(id);\n"
        "ALTER TABLE orders ADD FOREIGN KEY (region_id) REFERENCES regions(id);\n"
        "CREATE INDEX idx_orders_user ON orders (user_id);\n"
    ),
    "web/src/api/client.ts": '''import axios from 'axios';
import type { User } from '../types';
export const BASE = '/api';
export async function getUser(id: number): Promise<User> {
  const res = await fetch(`${BASE}/users/${id}`);
  return res.json();
}
export function listUsers() { return axios.get('/api/users'); }
export function createUser(u: User) { return fetch('/api/users', { method: 'POST', body: JSON.stringify(u) }); }
export const removeUser = (id: number) => axios.delete(`/api/users/${id}`);
export function ping() { return fetch('/py/health'); }
export function account(id: number) { return fetch(`/py/accounts/${id}`); }
export function unknown() { return fetch('/nowhere/at/all'); }
''',
    "web/src/types/index.ts": (
        "export interface User { id: number; name: string }\nexport type Id = number;\n"
    ),
    "web/src/components/UserCard.tsx": '''import React, { useEffect, useState } from 'react';
import { getUser, removeUser } from '../api/client';
import { BaseCard } from './BaseCard';
import type { User } from '../types';
import missing from './does-not-exist';

export class FancyCard extends BaseCard implements Renderable {}

export function UserCard({ id }: { id: number }) {
  const [user, setUser] = useState<User | null>(null);
  useEffect(() => { getUser(id).then(setUser); }, [id]);
  const onDelete = () => removeUser(id);
  return <div onClick={onDelete}>{user?.name}</div>;
}
''',
    "web/src/components/BaseCard.jsx": "export class BaseCard { render() { return null; } }\n",
    "web/src/index.js": (
        "import { UserCard } from './components/UserCard';\n"
        "const util = require('./util');\n"
        "export * from './api/client.js';\n"
    ),
    "web/src/util/index.js": "module.exports = { noop() {} };\n",
    "web/node_modules/pkg/index.js": "export const hidden = 1;\n",
    "vendor/autoload.php": "<?php class Hidden {}\n",
    "svc/app/__init__.py": "",
    "svc/app/main.py": '''from fastapi import FastAPI
from .db import get_conn
from . import models
from .models import Account
import svc.app.db as database

app = FastAPI()

class Special(Account):
    pass

@app.get("/py/health")
def health():
    conn = get_conn()
    conn.execute("SELECT count(*) FROM users")
    return {"ok": True}

@app.get("/py/accounts/{account_id}")
def account(account_id: int):
    return {}
''',
    "svc/app/db.py": "def get_conn():\n    return None\n",
    "svc/app/models.py": "class Account:\n    id: int\n",
}


# ======================================================================================
# Gate 1: Python Extraction Gate
# ======================================================================================


def test_python_imports_absolute_and_relative(tmp_path: Path) -> None:
    """Imports resolve to dotted module names; relative levels are applied to the package."""
    rec = _record(tmp_path, "backend/app/service.py", PY_SERVICE)
    assert rec.language == "python"
    assert rec.extractor == scout.EXTRACTOR_PY_AST
    assert rec.errors == ()
    imports = {e.target: e for e in _edges(rec, "import")}
    assert {"os", "json", "a.b.c", "fastapi.APIRouter"} <= set(imports)
    assert all(e.source == rec.path for e in imports.values())
    assert imports["json"].meta == {"alias": "j"}
    assert imports["backend.app.sibling"].meta == {
        "level": 1, "module": "backend.app", "name": "sibling",
    }
    assert imports["backend.app.models.User"].meta["module"] == "backend.app.models"
    assert imports["backend.app.models.Order"].meta["alias"] == "O"
    assert imports["backend.core.db.get_conn"].meta["level"] == 2
    assert imports["typing.Any"].meta == {"module": "typing", "name": "Any"}
    assert imports["os"].line == 3 and imports["backend.core.db.get_conn"].line == 7
    # __all__ becomes export edges
    assert _targets(rec, "export") == {"Service", "handler"}


def test_python_relative_import_from_package_init_and_escaping_root(tmp_path: Path) -> None:
    """``from .x`` inside ``pkg/__init__.py`` is relative to ``pkg`` itself."""
    rec = _record(tmp_path, "pkg/__init__.py", "from .mod import thing\nfrom . import other\n")
    assert _targets(rec, "import") == {"pkg.mod.thing", "pkg.other"}
    escaped = _record(tmp_path, "top.py", "from ... import far\n")
    (edge,) = _edges(escaped, "import")
    assert edge.target.startswith("...") and edge.target.endswith("far")
    assert edge.resolved_file is None


def test_python_classes_bases_decorators_and_nesting(tmp_path: Path) -> None:
    rec = _record(tmp_path, "backend/app/service.py", PY_SERVICE)
    service = _sym(rec, "Service", "class")
    assert service.signature == "class Service(Base, Generic[T], metaclass=Meta)"
    assert service.meta["bases"] == ["Base", "Generic[T]"]
    assert service.meta["decorators"] == ["dataclass(frozen=True, slots=True)"]
    assert "dataclass" in service.modifiers
    assert service.parent is None and service.language == "python"
    assert service.line == 27 and service.end_line > service.line
    assert _targets(rec, "extends", "Service") == {"Base", "Generic"}
    inner = _sym(rec, "Service.Inner", "class")
    assert inner.parent == "Service" and inner.name == "Inner"
    deep = _sym(rec, "Service.Inner.deep", "method")
    assert deep.parent == "Service.Inner"
    # class-level attributes
    assert _sym(rec, "Service.name", "property").signature == "name: str"
    assert _sym(rec, "Service.retries", "property").signature == "retries: int = 3"
    assert _sym(rec, "Service.LIMIT", "constant").parent == "Service"
    # enum detection
    assert _sym(rec, "Color", "enum").signature == "class Color(enum.Enum)"
    assert _names(rec, "enum_case") == ["Color.RED", "Color.GREEN"]


def test_python_function_and_method_signatures(tmp_path: Path) -> None:
    """Positional-only, keyword-only, defaults (elided when long), annotations, async."""
    rec = _record(tmp_path, "backend/app/service.py", PY_SERVICE)
    init = _sym(rec, "Service.__init__", "method")
    assert init.signature == (
        "def __init__(self, a, /, b: int = 5, *args: Any, key: str = ..., **kw: Any) -> None"
    )
    fetch = _sym(rec, "Service.fetch", "method")
    assert fetch.signature == (
        "async def fetch(conn, uid: int, *, timeout: float | None = None) -> dict[str, Any]"
    )
    assert fetch.modifiers == ("async", "staticmethod")
    assert fetch.meta["decorators"] == ["staticmethod"]
    assert fetch.parent == "Service"
    size = _sym(rec, "Service.size", "method")
    assert size.signature == "def size(self) -> int" and "property" in size.modifiers
    handler = _sym(rec, "handler", "function")
    assert handler.signature == "async def handler(user_id: int)"
    assert handler.meta["decorators"] == ["router.get('/users/{user_id}')"]
    assert handler.end_line == handler.line + 1
    nested = _sym(rec, "Service.fetch.nested", "function")
    assert nested.parent == "Service.fetch"


def test_python_constants_and_type_aliases(tmp_path: Path) -> None:
    rec = _record(tmp_path, "backend/app/service.py", PY_SERVICE)
    assert _sym(rec, "MAX_SIZE", "constant").signature == "MAX_SIZE = 10000"
    assert _sym(rec, "TIMEOUT", "constant").signature == "TIMEOUT: float = 3.5"
    assert _sym(rec, "T", "type_alias").signature == "T = TypeVar('T')"
    assert _sym(rec, "Effort", "type_alias").signature == "Effort = Literal['low', 'high']"
    assert _sym(rec, "Callback", "type_alias").signature == "Callback = Callable[[int], None]"
    assert _sym(rec, "Pair", "type_alias").signature == "type Pair[K, V] = tuple[K, V]"
    # every simple module-level binding is a symbol; lower-case ones are plain variables
    assert _sym(rec, "logger", "variable").signature == "logger = None"
    assert _sym(rec, "router", "variable").signature == "router = APIRouter(prefix='/api/v1')"
    assert "__all__" not in {s.name for s in rec.symbols}  # an export list, not a symbol


def test_python_call_edges_are_attributed_to_enclosing_scope(tmp_path: Path) -> None:
    rec = _record(tmp_path, "backend/app/service.py", PY_SERVICE)
    assert {"conn.cursor", "cur.execute", "conn.execute", "asyncio.gather", "nested"} <= _targets(
        rec, "call", "Service.fetch"
    )
    assert _targets(rec, "call", "Service.fetch.nested") == {"helper", "json.dumps"}
    assert _targets(rec, "call", "handler") == {"Service.fetch"}
    # module-level calls use the file path as source; builtins like len() are not edges
    assert "APIRouter" in _targets(rec, "call", rec.path)
    assert "len" not in _targets(rec, "call")
    # a class-body call belongs to the class
    assert _targets(rec, "call", "Service") == {"Manager"}


def test_python_docstring_first_line_only_and_clipped(tmp_path: Path) -> None:
    rec = _record(tmp_path, "backend/app/service.py", PY_SERVICE)
    assert _sym(rec, "Service").doc == "Service doc line."
    assert _sym(rec, "Service.fetch").doc == "Fetch one."
    assert _sym(rec, "handler").doc is None
    long_first = "word " * 100
    text = f'def f():\n    """{long_first}\n\n    second paragraph\n    """\n'
    doc = _sym(_record(tmp_path, "long_doc.py", text), "f").doc
    assert doc is not None and len(doc) <= MAX_DOC_CHARS and doc.endswith("...")
    assert "second" not in doc


def test_python_syntax_error_uses_fallback_and_records_error(tmp_path: Path) -> None:
    rec = _record(tmp_path, "pkg/broken.py", PY_BROKEN)
    assert rec.extractor == scout.EXTRACTOR_PY_FALLBACK
    assert len(rec.errors) == 1 and rec.errors[0].startswith("python parse error")
    assert "SyntaxError" in rec.errors[0]
    assert _sym(rec, "A", "class").signature == "class A(B)"
    assert _sym(rec, "A.m", "method").signature == "def m(self, x, y=1) -> int"
    assert _sym(rec, "A.n", "method").modifiers == ("async",)
    assert _sym(rec, "top", "function").line == 8
    imports = {e.target: e for e in _edges(rec, "import")}
    assert set(imports) == {"os", "pkg.x.a", "pkg.x.b"}
    assert imports["pkg.x.b"].meta["alias"] == "c"


def test_python_routes_and_embedded_sql(tmp_path: Path) -> None:
    """Decorator / urlpatterns routes and SQL passed to execute() are extracted too."""
    rec = _record(tmp_path, "backend/app/service.py", PY_SERVICE)
    routes = {s.name: s for s in rec.symbols if s.kind == "route"}
    assert set(routes) == {"GET /api/v1/users/{user_id}", "GET|POST /legacy", "ANY /items/<int:pk>"}
    assert routes["GET /api/v1/users/{user_id}"].meta["handler"] == "handler"
    assert routes["GET|POST /legacy"].meta["methods"] == ["GET", "POST"]
    assert routes["ANY /items/<int:pk>"].meta["framework"] == "django"
    queries = [s for s in rec.symbols if s.kind == "sql_query"]
    assert [q.meta["verb"] for q in queries] == ["SELECT", "SELECT", "DELETE"]
    assert queries[0].meta["tables"] == ["users", "orders"]
    assert queries[1].meta["tables"] == ["accounts"]  # resolved through the QUERY constant
    assert queries[2].signature == "sql execute: DELETE FROM {TABLE} WHERE id = {uid}"
    assert all(q.parent == "Service.fetch" for q in queries)
    assert _targets(rec, "sql_table", "Service.fetch") == {"users", "orders", "accounts"}


def test_python_pyi_stub_is_indexed(tmp_path: Path) -> None:
    rec = _record(tmp_path, "stubs/thing.pyi", "def f(x: int) -> str: ...\nclass C:\n    y: int\n")
    assert rec.language == "python"
    assert _sym(rec, "f").signature == "def f(x: int) -> str"
    assert _sym(rec, "C.y", "property").signature == "y: int"


# ======================================================================================
# Gate 2: PHP 8.3 Extraction Gate
# ======================================================================================

PHP_PATH = "app/Http/Controllers/UserController.php"
PHP_NS = "App\\Http\\Controllers"
PHP_CLASS = PHP_NS + "\\UserController"


def test_php_namespaces_braced_unbraced_and_multiple(tmp_path: Path) -> None:
    braced = _record(tmp_path, "ns/braced.php", PHP_NAMESPACES)
    assert braced.language == "php" and braced.extractor == "php-regex" and braced.errors == ()
    assert _names(braced, "namespace") == ["Alpha", "Beta\\Deep"]
    point = _sym(braced, "Alpha\\Point", "class")
    assert point.modifiers == ("readonly",) and point.signature == "readonly class Point"
    assert _sym(braced, "Alpha\\alpha_fn", "function").signature == "function alpha_fn(): void"
    assert _sym(braced, "Alpha\\VERSION", "constant").signature == "const VERSION = '1'"
    shape = _sym(braced, "Beta\\Deep\\Shape", "class")
    assert shape.modifiers == ("abstract",)
    area = _sym(braced, "Beta\\Deep\\Shape::area", "method")
    assert area.modifiers == ("abstract", "public") and area.parent == "Beta\\Deep\\Shape"
    assert _sym(braced, "Beta\\Deep\\Shape::name", "method").modifiers == ("final", "public")
    assert _names(braced, "enum_case") == [
        "Beta\\Deep\\Status::Active", "Beta\\Deep\\Status::Archived",
    ]
    unbraced = _record(
        tmp_path, "ns/unbraced.php",
        "<?php\nnamespace First;\nclass A {}\nnamespace Second;\nclass B extends \\First\\A {}\n"
        "class C extends A {}\n",
    )
    assert _names(unbraced, "namespace") == ["First", "Second"]
    assert _names(unbraced, "class") == ["First\\A", "Second\\B", "Second\\C"]
    assert _targets(unbraced, "extends", "Second\\B") == {"First\\A"}
    assert _targets(unbraced, "extends", "Second\\C") == {"Second\\A"}
    assert _sym(_record(tmp_path, "ns/none.php", "<?php\nclass Bare {}\n"), "Bare").parent is None


def test_php_use_imports_class_function_const_alias_group(tmp_path: Path) -> None:
    rec = _record(tmp_path, PHP_PATH, PHP_CONTROLLER)
    imports = {e.target: e for e in _edges(rec, "import")}
    assert set(imports) == {
        "App\\Models\\User",
        "App\\Repositories\\UserRepository",
        "App\\Repositories\\OrderRepository",
        "App\\Support\\helper",
        "App\\Support\\VERSION",
        "Symfony\\Component\\Routing\\Annotation\\Route",
    }
    assert imports["App\\Models\\User"].meta == {}
    assert imports["App\\Repositories\\OrderRepository"].meta == {"alias": "Orders"}
    assert imports["App\\Repositories\\UserRepository"].line == 7  # group use
    assert imports["App\\Support\\helper"].meta == {"alias": "h", "kind": "function"}
    assert imports["App\\Support\\VERSION"].meta == {"kind": "const"}


def test_php_classes_interfaces_traits_enums(tmp_path: Path) -> None:
    rec = _record(tmp_path, PHP_PATH, PHP_CONTROLLER)
    cls = _sym(rec, PHP_CLASS, "class")
    assert cls.signature == (
        "final class UserController extends BaseController implements Jsonable, \\Countable"
    )
    assert cls.modifiers == ("final",) and cls.doc == "Handles users."
    assert cls.meta["attributes"] == ["Route"]
    assert cls.line == 17 and cls.end_line == 66
    assert _targets(rec, "extends", PHP_CLASS) == {PHP_NS + "\\BaseController"}
    assert _targets(rec, "implements", PHP_CLASS) == {PHP_NS + "\\Jsonable", "Countable"}
    assert _targets(rec, "uses_trait", PHP_CLASS) == {
        PHP_NS + "\\LogsActivity", PHP_NS + "\\Cacheable",
    }
    iface = _sym(rec, PHP_NS + "\\Jsonable", "interface")
    assert iface.signature == "interface Jsonable extends \\JsonSerializable, Arrayable"
    assert _targets(rec, "extends", iface.qualified_name) == {
        "JsonSerializable", PHP_NS + "\\Arrayable",
    }
    assert _sym(rec, PHP_NS + "\\Jsonable::toJson", "method").signature == (
        "public function toJson(int $flags = 0): string"
    )
    assert _sym(rec, PHP_NS + "\\LogsActivity", "trait").signature == "trait LogsActivity"
    suit = _sym(rec, PHP_NS + "\\Suit", "enum")
    assert suit.signature == "enum Suit: string implements HasLabel"
    assert suit.meta["backing_type"] == "string"
    assert _targets(rec, "implements", suit.qualified_name) == {PHP_NS + "\\HasLabel"}
    cases = [s for s in rec.symbols if s.kind == "enum_case"]
    assert [c.signature for c in cases] == ["case Hearts = 'H'", "case Spades = 'S'"]
    assert all(c.parent == suit.qualified_name for c in cases)
    assert _sym(rec, PHP_NS + "\\Suit::label", "method").meta["returns"] == "string"
    # typed and multi-declaration class constants
    assert _sym(rec, PHP_CLASS + "::TABLE", "constant").signature == (
        "public const string TABLE = 'users'"
    )
    assert _sym(rec, PHP_CLASS + "::A", "constant").modifiers == ("private",)
    assert _sym(rec, PHP_CLASS + "::B", "constant").signature == "private const B = [1, 2]"
    assert _sym(rec, PHP_NS + "\\Suit::Wild", "constant").parent == suit.qualified_name


def test_php_method_signatures_and_properties(tmp_path: Path) -> None:
    """Visibility, static, abstract, final, by-ref, variadic, promoted ctor props, PHP 8 types."""
    rec = _record(tmp_path, PHP_PATH, PHP_CONTROLLER)
    ctor = _sym(rec, PHP_CLASS + "::__construct", "method")
    assert ctor.signature == (
        "public function __construct(private readonly \\PDO $pdo, "
        "protected ?Orders $orders = null, int $plain = 5)"
    )
    pdo = _sym(rec, PHP_CLASS + "::$pdo", "property")
    assert pdo.signature == "private readonly \\PDO $pdo"
    assert pdo.modifiers == ("private", "readonly", "promoted")
    assert _sym(rec, PHP_CLASS + "::$orders", "property").modifiers == ("protected", "promoted")
    assert PHP_CLASS + "::$plain" not in _names(rec, "property")  # not promoted
    show = _sym(rec, PHP_CLASS + "::show", "method")
    assert show.signature == "public function show(int $id, string ...$rest): ?User"
    assert show.meta == {"attributes": ["Route"], "returns": "?User", "variadic": True}
    assert show.doc == "Show one user." and show.parent == PHP_CLASS
    assert show.end_line > show.line
    make = _sym(rec, PHP_CLASS + "::make", "method")
    assert make.signature == (
        "abstract protected static function &make(array &$data, (A&B)|null $dnf = null): static"
    )
    assert make.modifiers == ("abstract", "protected", "static", "by-ref")
    helper = _sym(rec, PHP_CLASS + "::helper", "method")
    assert helper.modifiers == ("final", "public", "static") and helper.meta["returns"] == "void"
    # declared properties: nullable, union, static, readonly
    assert _sym(rec, PHP_CLASS + "::$repo", "property").signature == (
        "protected static ?UserRepository $repo"
    )
    assert _sym(rec, PHP_CLASS + "::$id", "property").signature == "public readonly int|string $id"


def test_php_top_level_functions_constants_and_attributes(tmp_path: Path) -> None:
    rec = _record(tmp_path, PHP_PATH, PHP_CONTROLLER)
    fn = _sym(rec, PHP_NS + "\\top_level", "function")
    assert fn.signature == "function top_level(int $a, $b = 'x)y'): int"
    assert fn.parent is None and fn.meta["returns"] == "int"
    assert _targets(rec, "call", fn.qualified_name) == {"other_fn"}
    assert _sym(rec, "APP_ROOT", "constant").signature == "define('APP_ROOT', ...)"
    assert _sym(rec, PHP_NS + "\\GLOBAL_C", "constant").signature == "const GLOBAL_C = 3"
    attributed = _record(
        tmp_path, "attr.php",
        "<?php\n#[Deprecated, Pure]\n#[\\App\\Attr\\Cached(ttl: 60)]\nfunction cached(): int { return 1; }\n",
    )
    assert _sym(attributed, "cached").meta["attributes"] == [
        "Deprecated", "Pure", "\\App\\Attr\\Cached",
    ]


def test_php_pdo_sql_prepare_query_exec(tmp_path: Path) -> None:
    """Statement verb, tables and a clipped normalized preview for every quoting style."""
    rec = _record(tmp_path, PHP_PATH, PHP_CONTROLLER)
    queries = [s for s in rec.symbols if s.kind == "sql_query"]
    assert [(q.meta["api"], q.meta["verb"], q.meta["tables"]) for q in queries] == [
        ("prepare", "SELECT", ["users", "orders"]),  # single quoted
        ("exec", "UPDATE", ["audit_log"]),  # double quoted + concatenation through $sql
        ("query", "SELECT", ["app.sessions"]),  # heredoc
        ("exec", "DELETE", ["sessions"]),  # nowdoc
        ("query", "SELECT", ["roles"]),  # double quoted, backticked table
    ]
    assert queries[0].signature == (
        "sql prepare: SELECT u.*, o.total FROM users u JOIN orders o ON o.user_id = u.id "
        "WHERE u.id = ?"
    )
    assert queries[1].signature == "sql exec: UPDATE `audit_log` SET seen = 1 WHERE user_id = {$id}"
    assert queries[2].signature == "sql query: SELECT * FROM app.sessions WHERE user_id = {$id}"
    assert queries[3].signature == "sql exec: DELETE FROM sessions WHERE expired = 1"
    assert queries[0].parent == PHP_CLASS + "::show" and queries[3].parent == PHP_CLASS + "::purge"
    edges = {(e.source, e.target, e.meta["verb"]) for e in _edges(rec, "sql_table")}
    assert edges == {
        (PHP_CLASS + "::show", "users", "SELECT"),
        (PHP_CLASS + "::show", "orders", "SELECT"),
        (PHP_CLASS + "::show", "audit_log", "UPDATE"),
        (PHP_CLASS + "::show", "app.sessions", "SELECT"),
        (PHP_CLASS + "::purge", "sessions", "DELETE"),
        (PHP_CLASS + "::purge", "roles", "SELECT"),
    }
    long_sql = "SELECT " + ", ".join(f"col_{i}" for i in range(200)) + " FROM wide_table"
    wide = _record(tmp_path, "wide.php", f"<?php\n$pdo->query('{long_sql}');\n")
    (query,) = [s for s in wide.symbols if s.kind == "sql_query"]
    assert query.meta["tables"] == ["wide_table"]
    assert len(query.signature) <= len("sql query: ") + MAX_SQL_PREVIEW_CHARS
    assert query.signature.endswith("...")


def test_php_routes_laravel_slim_symfony(tmp_path: Path) -> None:
    rec = _record(tmp_path, "routes/api.php", PHP_ROUTES)
    routes = {s.name: s for s in rec.symbols if s.kind == "route"}
    ctl = "App\\Http\\UserController"
    assert routes["GET /api/users"].meta == {
        "framework": "laravel", "handler": ctl + "::index", "methods": ["GET"],
        "name": "users.index", "path": "/api/users",
    }
    assert routes["GET /api/users/{id}"].meta["handler"] == ctl + "::show"
    assert routes["POST /api/users"].meta["handler"] == ctl + "::store"  # 'Class@method' string
    assert routes["PUT|PATCH /api/users/{id}"].meta["methods"] == ["PUT", "PATCH"]  # Route::match
    assert routes["DELETE /api/users/{id}"].signature == (
        "ROUTE DELETE /api/users/{id} -> App\\Http\\UserController::destroy"
    )
    # apiResource: five routes; resource: seven (adds create / edit)
    orders = sorted(name for name in routes if "/api/orders" in name)
    assert orders == [
        "DELETE /api/orders/{id}", "GET /api/orders", "GET /api/orders/{id}",
        "POST /api/orders", "PUT|PATCH /api/orders/{id}",  # update answers both verbs
    ]
    assert routes["GET /api/orders"].meta["handler"] == "OrderController::index"
    photos = sorted(name for name in routes if "/api/photos" in name)
    assert len(photos) == 7 and "GET /api/photos/create" in photos
    assert routes["GET /api/photos/{id}/edit"].meta["handler"] == "PhotoController::edit"
    # Slim / Lumen style
    assert routes["GET /slim/ping"].meta["framework"] == "slim"
    assert routes["GET /slim/ping"].meta["handler"] == ctl + "::index"
    assert routes["POST /slim/items"].meta["handler"] == "closure"
    route_edges = {(e.source, e.target) for e in _edges(rec, "route")}
    assert ("GET /api/users", ctl + "::index") in route_edges
    assert not any(target == "closure" for _, target in route_edges)
    lumen = _record(
        tmp_path, "routes/web.php",
        "<?php\n$router->get('/lumen/items', 'ItemController@index');\n"
        "$router->group(['prefix' => 'admin'], function () use ($router) {\n"
        "    $router->post('/items', ['uses' => 'Admin\\ItemController@store']);\n"
        "    $router->get('settings', 'SettingsController@show');\n"
        "});\n"
        "$value = $app->get('config');\n",  # a container lookup, not a route
    )
    assert {s.name: s.meta["handler"] for s in lumen.symbols if s.kind == "route"} == {
        "GET /lumen/items": "ItemController::index",
        "POST /admin/items": "Admin\\ItemController::store",
        "GET /admin/settings": "SettingsController::show",
    }
    # Symfony attributes: class-level prefix + method-level path / methods / name
    symfony = _record(tmp_path, PHP_PATH, PHP_CONTROLLER)
    sroutes = {s.name: s for s in symfony.symbols if s.kind == "route"}
    attr = sroutes["GET|HEAD /users/{id}"]
    assert attr.meta == {
        "framework": "symfony", "handler": PHP_CLASS + "::show", "methods": ["GET", "HEAD"],
        "name": "show", "path": "/users/{id}",
    }
    assert attr.parent == PHP_CLASS and attr.line == 33
    assert sroutes["GET /health"].meta["handler"] == "closure"  # Route::get after ?> ... <?php


def test_php_call_edges(tmp_path: Path) -> None:
    rec = _record(tmp_path, PHP_PATH, PHP_CONTROLLER)
    show = PHP_CLASS + "::show"
    calls = {e.target: e for e in _edges(rec, "call", show)}
    assert calls["App\\Models\\User"].meta == {"via": "new"}  # new X through the use alias
    assert "App\\Models\\User::find" in calls  # X::method()
    assert "App\\Support\\helper" in calls  # function call through `use function ... as h`
    assert "strlen" in calls  # plain function call
    assert PHP_CLASS + "::helper" in calls  # self::
    assert PHP_NS + "\\BaseController::boot" in calls  # parent::
    assert PHP_CLASS + "::log" in calls  # $this->method()
    assert {"->prepare", "->execute", "->refresh"} <= set(calls)  # ->method() best effort
    assert "error_log" in _targets(rec, "call", PHP_NS + "\\LogsActivity::log")


def test_php_masking_ignores_strings_comments_and_inline_html(tmp_path: Path) -> None:
    rec = _record(tmp_path, PHP_PATH, PHP_CONTROLLER)
    names = {s.name for s in rec.symbols}
    assert not {"Fake", "nope", "Ghost", "ghost"} & names
    assert not any("commented_out" in t for t in _targets(rec, "sql_table"))
    assert _record(tmp_path, "tpl/view.phtml", "<p>class X {}</p>\n").symbols == ()
    inc = _record(tmp_path, "lib/legacy.inc", "<?php function legacy_helper($a) { return $a; }\n")
    assert inc.language == "php" and _names(inc, "function") == ["legacy_helper"]


# ======================================================================================
# Gate 3: JavaScript / TypeScript Extraction Gate
# ======================================================================================

TSX_PATH = "src/App.tsx"


def test_js_ts_extensions_are_recognised(tmp_path: Path) -> None:
    sources = {
        "a.js": "export function f1() {}\n",
        "b.jsx": "export function f2() { return <b/>; }\n",
        "c.mjs": "export function f3() {}\n",
        "d.cjs": "function f4() {}\nmodule.exports = { f4 };\n",
        "e.ts": "export function f5(x: number): number { return x; }\n",
        "f.tsx": "export function f6(): JSX.Element { return <i/>; }\n",
        "g.mts": "export function f7(): void {}\n",
        "h.cts": "export function f8(): void {}\n",
        "i.d.ts": "export declare function f9(x: number): string;\n",
    }
    index = _index(_write(tmp_path, sources))
    languages = {rec.path: rec.language for rec in index.files}
    assert languages == {
        "a.js": "javascript", "b.jsx": "javascript", "c.mjs": "javascript",
        "d.cjs": "javascript", "e.ts": "typescript", "f.tsx": "typescript",
        "g.mts": "typescript", "h.cts": "typescript", "i.d.ts": "typescript",
    }
    for i, rec in enumerate(index.files, start=1):
        assert rec.errors == () and rec.extractor == f"{rec.language}-regex"
        assert index.find_symbol(f"f{i}", path=rec.path), rec.path
    assert index.find_symbol("f9")[0].signature == (
        "export declare function f9(x: number): string"
    )
    assert language_for_path("x/y/thing.D.TS") == "typescript"
    assert language_for_path("README.md") is None
    assert set(LANGUAGE_BY_EXTENSION.values()) == {
        "python", "php", "javascript", "typescript", "sql",
    }


def test_js_import_forms(tmp_path: Path) -> None:
    rec = _record(tmp_path, TSX_PATH, TSX_APP)
    assert rec.language == "typescript" and rec.extractor == "typescript-regex"
    assert rec.errors == ()
    imports = {e.target: e for e in _edges(rec, "import")}
    assert imports["react"].meta == {  # default + named (with alias)
        "default": "React", "names": ["useState", "useEffect as useFx"],
    }
    assert imports["./api/client"].meta == {"namespace": "api"}
    assert imports["../types"].meta == {"names": ["User", "Role"], "type_only": True}
    assert imports["./thing.js"].meta == {"default": "DefaultThing"}
    assert imports["./styles.css"].meta == {"side_effect": True}
    assert imports["legacy-lib"].meta == {"default": "legacy", "require": True}
    assert imports["./util"].meta == {"names": ["a", "b: c"], "require": True}
    assert imports["./lazy/Mod"].meta == {"dynamic": True}
    assert imports["./lazy/Mod"].line == 9 and imports["react"].line == 2
    assert all(e.source == TSX_PATH for e in imports.values())


def test_js_export_forms(tmp_path: Path) -> None:
    rec = _record(tmp_path, TSX_PATH, TSX_APP)
    exports = _edges(rec, "export")
    by_target: dict[str, list[Edge]] = {}
    for edge in exports:
        by_target.setdefault(edge.target, []).append(edge)
    assert by_target["helper"][0].meta == {"from": "./helpers"}  # re-export
    assert by_target["renamed"][0].meta == {"from": "./helpers", "local": "other"}
    assert by_target["*"][0].meta == {"from": "./all"}  # export * from
    assert by_target["ns"][0].meta == {"from": "./ns"}  # export * as ns from
    for named in ("MAX_ITEMS", "CardProps", "Id", "Color", "Repo", "Card", "Row", "useThing", "gen"):
        assert named in by_target, named  # export const / interface / type / enum / class / fn
    defaults = by_target["default"]
    assert {e.line for e in defaults} == {59, 85}
    assert defaults[-1].meta == {"local": "App", "style": "export="}  # export =
    commonjs = {e.target for e in exports if e.meta.get("style") == "commonjs"}
    assert commonjs == {"plain", "square", "extra"}
    plain_default = _record(tmp_path, "d.js", "const x = 1;\nexport default x;\nexport { x as y };\n")
    assert _edge(plain_default, "export", "default").meta == {"local": "x"}
    assert _edge(plain_default, "export", "y").meta == {"local": "x"}


def test_js_classes_and_members(tmp_path: Path) -> None:
    rec = _record(tmp_path, TSX_PATH, TSX_APP)
    repo = _sym(rec, "Repo", "class")
    assert repo.signature == (
        "export abstract class Repo<T> extends Base<T> implements Readable, Writable<T>"
    )
    assert repo.modifiers == ("export", "abstract")
    assert repo.line == 32 and repo.end_line == 57
    assert _targets(rec, "extends", "Repo") == {"Base"}
    assert _targets(rec, "implements", "Repo") == {"Readable", "Writable"}
    methods = {s.signature: s for s in rec.symbols if s.kind == "method" and s.parent == "Repo"}
    assert set(methods) == {
        "constructor(private http: HttpClient, name: string)",
        "get size(): number",
        "set size(v: number)",
        "static create<T>(name: string): Repo<T>",
        "async *stream(): AsyncGenerator<T>",
        "abstract find(id: Id): Promise<T | undefined>",
        "handleClick = async (e: MouseEvent): Promise<void> =>",
        "protected load(id: Id): void",
    }
    assert methods["get size(): number"].modifiers == ("getter",)
    assert methods["set size(v: number)"].modifiers == ("setter",)
    assert methods["static create<T>(name: string): Repo<T>"].modifiers == ("static",)
    assert methods["async *stream(): AsyncGenerator<T>"].modifiers == ("async", "generator")
    assert methods["abstract find(id: Id): Promise<T | undefined>"].modifiers == ("abstract",)
    arrow = methods["handleClick = async (e: MouseEvent): Promise<void> =>"]
    assert arrow.modifiers == ("arrow", "async") and arrow.qualified_name == "Repo.handleClick"
    assert arrow.end_line == arrow.line + 3
    props = {s.name: s for s in rec.symbols if s.kind == "property" and s.parent == "Repo"}
    assert set(props) == {"instances", "cache", "#secret"}
    assert props["cache"].signature == "private readonly cache: Map<string, T>"
    assert props["instances"].modifiers == ("static",)
    # call edges are attributed to the enclosing member
    assert "super" in _targets(rec, "call", "Repo.constructor")
    assert _edge(rec, "call", "ConcreteRepo").meta == {"new": True}
    assert _edge(rec, "call", "ConcreteRepo").source == "Repo.create"
    react_class = _record(
        tmp_path, "Legacy.jsx",
        "class Legacy extends React.Component {\n  render() { return <div/>; }\n}\n",
    )
    assert "component" in _sym(react_class, "Legacy", "class").modifiers


def test_js_functions_declarations_and_bound_expressions(tmp_path: Path) -> None:
    rec = _record(tmp_path, TSX_PATH, TSX_APP)
    gen = _sym(rec, "gen", "function")
    assert gen.signature == (
        "export async function* gen<T extends object = {}>(a: T, b = 2): AsyncGenerator<T>"
    )
    assert gen.modifiers == ("export", "async", "generator")
    assert _sym(rec, "plain", "function").signature == "function plain(a, b)"
    assert _targets(rec, "call", "plain") == {"helperFn"}
    square = _sym(rec, "square", "function")  # const bound arrow, single parameter
    assert square.signature == "const square = (x) =>" and square.modifiers == ("arrow",)
    legacy = _sym(rec, "legacyFn", "function")  # var bound function expression
    assert legacy.signature == "var legacyFn = function (q)"
    assert legacy.modifiers == ("function-expression",)
    lazy = _sym(rec, "lazyMod", "function")
    assert lazy.signature == "const lazyMod = () =>"
    assert _sym(rec, "extra", "function").modifiers == ("export", "function-expression")
    draw = _sym(rec, "Widget.draw", "method")  # prototype assignment
    assert draw.parent == "Widget" and _targets(rec, "call", "Widget.draw") == {"ctx.fill"}
    nested = _sym(rec, "App.handleSave", "function")
    assert nested.parent == "App" and _targets(rec, "call", "App.handleSave") == {"save"}
    let_bound = _record(tmp_path, "l.js", "let later = async (a, b) => { await go(a); };\n")
    assert _sym(let_bound, "later").signature == "let later = async (a, b) =>"
    assert _sym(let_bound, "later").modifiers == ("async", "arrow")
    # plain values are constants / variables
    assert _sym(rec, "API_BASE", "constant").signature == "const API_BASE = '/api/v1'"
    assert _sym(rec, "MAX_ITEMS", "constant").signature == "export const MAX_ITEMS: number"
    assert _sym(rec, "counter", "variable").signature == "let counter"


def test_ts_interfaces_type_aliases_enums_namespaces(tmp_path: Path) -> None:
    rec = _record(tmp_path, TSX_PATH, TSX_APP)
    props = _sym(rec, "CardProps", "interface")
    assert props.signature == "export interface CardProps extends BaseProps<User>, Clickable"
    assert props.doc == "A user card." and props.end_line == props.line + 4
    assert _targets(rec, "extends", "CardProps") == {"BaseProps", "Clickable"}
    assert _sym(rec, "CardProps.user", "property").signature == "user: User"
    assert _sym(rec, "CardProps.onSelect", "property").signature == (
        "onSelect?: (id: string) => void"
    )
    render = _sym(rec, "CardProps.render", "method")
    assert render.signature == "render(): JSX.Element" and render.modifiers == ("signature",)
    assert _sym(rec, "Id", "type_alias").signature == "export type Id = string | number"
    assert _sym(rec, "Handler", "type_alias").signature == "type Handler<T> = (value: T) => void"
    assert _sym(rec, "Color", "enum").signature == "export enum Color"
    cases = [s for s in rec.symbols if s.kind == "enum_case"]
    assert [(c.qualified_name, c.signature) for c in cases] == [
        ("Color.Red", "Red"), ("Color.Green", "Green = 'green'"), ("Color.Blue", "Blue = 4"),
    ]
    namespace = _sym(rec, "Legacy.Stuff", "namespace")
    assert namespace.modifiers == ("declare",)
    assert _sym(rec, "Legacy.Stuff.old", "function").parent == "Legacy.Stuff"
    # none of this exists in plain JavaScript mode
    js = _record(tmp_path, "types.js", "const type = 1;\nvar interface_ = 2;\n")
    assert not [s for s in js.symbols if s.kind in ("interface", "type_alias", "enum")]


def test_react_components_and_hook_usage(tmp_path: Path) -> None:
    rec = _record(tmp_path, TSX_PATH, TSX_APP)
    app = _sym(rec, "App", "component")
    assert app.signature == "export default function App({ items }: { items: string[] })"
    assert app.meta["hooks"] == ["useState", "useFx", "useRef"]
    assert _targets(rec, "uses_hook", "App") == {"useState", "useFx", "useRef"}
    card = _sym(rec, "Card", "component")  # React.memo(function ...)
    assert card.meta["wrapper"] == "memo"
    assert _sym(rec, "Row", "component").signature == (
        "export const Row: React.FC<CardProps> = ({ user }) =>"
    )
    hook = _sym(rec, "useThing", "hook")
    assert hook.meta["hooks"] == ["useState"]
    assert _targets(rec, "uses_hook", "useThing") == {"useState"}
    # an upper-case function without JSX is not a component; JSX in .js files counts
    assert _sym(_record(tmp_path, "util.ts", "export function Build() { return 1; }\n"),
                "Build").kind == "function"
    assert _sym(_record(tmp_path, "W.js", "export const W = () => <div/>;\n"), "W").kind == (
        "component"
    )


def test_jsx_and_dom_event_handlers(tmp_path: Path) -> None:
    rec = _record(tmp_path, TSX_PATH, TSX_APP)
    handlers = {(e.meta["via"], e.meta["event"]): e for e in _edges(rec, "handler")}
    assert handlers[("jsx", "onClick")].target == "handleSave"
    assert handlers[("jsx", "onMouseEnter")].target == "setN"  # inline arrow -> first call
    assert handlers[("jsx", "onSelect")].target == "this.props.onPick"
    assert handlers[("jsx", "onChange")].target == "track"
    assert handlers[("addEventListener", "click")].target == "handleSave"
    assert handlers[("property", "resize")].target == "relayout"  # window.onresize = function
    assert handlers[("property", "click")].target == "handleSave"  # btn.onclick = handleSave
    assert all(e.source == "App" for e in handlers.values())
    # JSX text with an apostrophe ("Don't panic") must not swallow the rest of the file
    assert _sym(rec, "useThing").line == 77


def test_js_fetch_axios_xhr_jquery_api_calls(tmp_path: Path) -> None:
    """HTTP method and URL for every client style; template literals become placeholders."""
    rec = _record(tmp_path, TSX_PATH, TSX_APP)
    calls = {e.target: e for e in _edges(rec, "api_call")}
    assert set(calls) == {
        "DELETE /api/v1/users/{e.detail}/items?x=1",  # fetch + options.method + template
        "POST /api/users",  # this.http.post<User>(...)
        "GET /api/items/{id}/detail",  # axios.get with string concatenation
        "PUT /api/raw",  # axios({ url, method })
        "PATCH /api/v2/things/{id}",  # axios.create({ baseURL }) instance
        "POST /api/legacy",  # $.ajax({ url, type })
        "GET /api/json",  # $.getJSON
        "POST /api/form",  # $.post
        "GET /api/xhr/{id}",  # XMLHttpRequest.open
        "ANY {url}",  # fetch(url, opts): dynamic
    }
    delete = calls["DELETE /api/v1/users/{e.detail}/items?x=1"]
    assert delete.meta == {
        "client": "fetch", "method": "DELETE", "url": "/api/v1/users/{e.detail}/items?x=1",
    }
    assert delete.source == "Repo.handleClick"
    assert calls["PUT /api/raw"].meta["client"] == "axios"
    assert calls["POST /api/legacy"].meta["client"] == "jquery"
    assert calls["GET /api/xhr/{id}"].meta["client"] == "xhr"
    assert calls["GET /api/xhr/{id}"].source == "Repo.load"
    simple = _record(
        tmp_path, "simple.js",
        "async function go() {\n"
        "  await fetch('/a');\n"
        "  await fetch('/b', { method: 'put' });\n"
        "  await axios.post(`/c/${x}/d`, body);\n"
        "  await axios.request({ method: 'delete', url: '/e' });\n"
        "  $.get('/f');\n"
        "  jQuery.ajax('/g', { method: 'PATCH' });\n"
        "}\n",
    )
    assert _targets(simple, "api_call", "go") == {
        "GET /a", "PUT /b", "POST /c/{x}/d", "DELETE /e", "GET /f", "PATCH /g",
    }


def test_js_masking_of_strings_comments_regex_and_templates(tmp_path: Path) -> None:
    text = (
        "// function commented() {}\n"
        "/* class Hidden {} */\n"
        "const s = 'function inString() {}';\n"
        "const t = `class InTemplate { ${real()} }`;\n"
        "const r = /function inRegex\\(\\) {}/g;\n"
        "function actual() { return 1; }\n"
    )
    rec = _record(tmp_path, "mask.js", text)
    assert _names(rec, "function") == ["actual"] and _names(rec, "class") == []
    assert _targets(rec, "call") == {"real"}  # code inside ${...} is still code


# ======================================================================================
# Gate 4: SQL Extraction Gate
# ======================================================================================


def test_sql_create_table_columns_keys_and_foreign_keys(tmp_path: Path) -> None:
    rec = _record(tmp_path, SQL_PATH, SQL_SCHEMA)
    assert rec.language == "sql" and rec.extractor == "sql-regex" and rec.errors == ()
    users = _sym(rec, "users", "table")
    assert users.meta == {
        "column_count": 4,
        "columns": ["id INT UNSIGNED PK", "email VARCHAR(255)", "role ENUM('admin','user')",
                    "key VARCHAR(50)"],
        "primary_key": ["id"],
    }
    assert users.signature.startswith("TABLE users (id INT UNSIGNED PK, email VARCHAR(255)")
    assert users.line == 3 and users.end_line == 11
    orders = _sym(rec, "public.orders", "table")
    assert orders.name == "orders" and orders.meta["primary_key"] == ["id"]
    assert orders.meta["columns"] == [
        "id BIGSERIAL PK", "user_id INTEGER", "total NUMERIC(10, 2)",
        "created_at TIMESTAMP WITH TIME ZONE",
    ]
    fks = [e for e in _edges(rec, "foreign_key") if e.source == "public.orders"]
    inline, composite = fks
    assert inline.target == "users" and inline.meta == {  # inline REFERENCES
        "columns": ["user_id"], "on_delete": "CASCADE", "ref_columns": ["id"],
    }
    assert composite.target == "geo.zones" and composite.meta == {  # table-level, composite
        "columns": ["ship_country", "ship_zip"], "constraint": "fk_ship",
        "on_delete": "SET NULL", "on_update": "NO ACTION", "ref_columns": ["country", "zip"],
    }
    # table-level UNIQUE KEY / KEY become indexes
    assert _sym(rec, "uq_email", "index").meta == {
        "columns": ["email"], "table": "users", "unique": True,
    }
    assert _sym(rec, "idx_role", "index").parent == "users"


def test_sql_alter_table(tmp_path: Path) -> None:
    rec = _record(tmp_path, SQL_PATH, SQL_SCHEMA)
    note = _sym(rec, "orders.note", "column")  # ADD COLUMN
    assert note.signature == "ALTER TABLE orders ADD COLUMN note TEXT"
    assert note.meta == {"table": "orders", "type": "TEXT"}
    fks = {e.target: e for e in _edges(rec, "foreign_key") if e.source == "orders"}
    assert fks["users"].meta == {  # ADD CONSTRAINT ... FOREIGN KEY
        "columns": ["user_id"], "constraint": "fk_orders_user", "on_update": "CASCADE",
        "ref_columns": ["id"],
    }
    assert fks["coupons"].meta == {"columns": ["coupon_id"], "ref_columns": ["id"]}  # ADD FK
    assert _sym(rec, "idx_total", "index").signature == "INDEX idx_total ON orders (total)"
    assert _sym(rec, "uq_note", "index").meta["unique"] is True  # ADD UNIQUE INDEX
    assert {"ALTER"} == {e.meta["verb"] for e in _edges(rec, "sql_table") if e.line in (23, 24, 25)
                         and e.source == rec.path}


def test_sql_create_index_view_and_drop(tmp_path: Path) -> None:
    rec = _record(tmp_path, SQL_PATH, SQL_SCHEMA)
    index = _sym(rec, "idx_orders_user", "index")
    assert index.signature == (
        "UNIQUE INDEX idx_orders_user ON public.orders USING btree (user_id, lower(note)) "
        "WHERE total > 0"
    )
    assert index.meta == {
        "columns": ["user_id", "lower(note)"], "concurrently": True, "partial": True,
        "table": "public.orders", "unique": True, "using": "btree", "where": "total > 0",
    }
    unnamed = _sym(rec, "orders_created_at_idx", "index")  # CREATE INDEX ON ... (no name)
    assert unnamed.meta == {"columns": ["created_at"], "table": "orders", "unique": False}
    assert _sym(rec, "idx_my", "index").meta["using"] == "HASH"
    view = _sym(rec, "active_users", "view")
    assert view.signature == "VIEW active_users (from users, orders)"
    assert view.meta == {"materialized": False, "tables": ["users", "orders"]}
    assert _targets(rec, "sql_table", "active_users") == {"users", "orders"}
    drops = {e.target for e in _edges(rec, "sql_table") if e.meta["verb"] == "DROP"}
    assert drops == {"legacy_a", "legacy_b"}
    assert _sym(rec, "trg_touch", "trigger").meta == {"table": "orders"}
    extra = _record(
        tmp_path, "extra.sql",
        "CREATE MATERIALIZED VIEW mv AS SELECT * FROM t1;\n"
        "CREATE UNIQUE INDEX IF NOT EXISTS ux ON t1 (a, b);\n",
    )
    assert _sym(extra, "mv", "view").meta["materialized"] is True
    assert _sym(extra, "ux", "index").meta == {"columns": ["a", "b"], "table": "t1", "unique": True}


def test_sql_dialect_tolerance(tmp_path: Path) -> None:
    """Backticks, double quotes, schema-qualified names, brackets, $$ bodies, DELIMITER."""
    rec = _record(tmp_path, SQL_PATH, SQL_SCHEMA)
    assert _names(rec, "table") == ["users", "public.orders", "dbo.Audit Log"]
    audit = _sym(rec, "dbo.Audit Log", "table")
    assert audit.meta["columns"] == ["Id INT PK", "UserId INT"]
    (bracket_fk,) = [e for e in _edges(rec, "foreign_key") if e.source == "dbo.Audit Log"]
    assert bracket_fk.target == "dbo.Users" and bracket_fk.meta["ref_columns"] == ["Id"]
    # statements inside a dollar-quoted body or a DELIMITER block are not top-level statements
    all_targets = _targets(rec, "sql_table")
    assert "ghost" not in all_targets and "never_seen" not in all_targets
    assert "sessions" not in all_targets
    assert _sym(rec, "touch", "function").end_line == 36
    assert _sym(rec, "cleanup", "procedure").line == 39
    # a semicolon inside a string / comment never splits a statement
    insert = [e for e in _edges(rec, "sql_table") if e.meta["verb"] == "INSERT"]
    assert [(e.target, e.line) for e in insert] == [("users", 45)]
    assert "email VARCHAR(255)" in _sym(rec, "users", "table").meta["columns"]


def test_sql_migration_detection_and_operations(tmp_path: Path) -> None:
    rec = _record(tmp_path, SQL_PATH, SQL_SCHEMA)
    (migration,) = [s for s in rec.symbols if s.kind == "migration"]
    assert migration.qualified_name == "migration:2024_01_15_120000"
    assert migration.name == "create_core"
    assert migration.meta["version"] == "2024_01_15_120000"
    assert migration.meta["direction"] == "up" and migration.meta["style"] == "timestamp"
    assert migration.meta["operations"] == [
        "CREATE TABLE users",
        "CREATE TABLE public.orders",
        "CREATE TABLE dbo.Audit Log",
        "ALTER TABLE orders ADD COLUMN note",
        "ALTER TABLE orders ADD FOREIGN KEY (user_id) -> users",
        "ALTER TABLE orders ADD FOREIGN KEY (coupon_id) -> coupons",
        "ALTER TABLE orders ADD INDEX idx_total (total)",
        "ALTER TABLE orders ADD UNIQUE INDEX uq_note (note)",
        "ALTER TABLE orders DROP COLUMN legacy",
        "CREATE UNIQUE INDEX idx_orders_user ON public.orders",
        "CREATE INDEX orders_created_at_idx ON orders",
        "CREATE INDEX idx_my ON users",
        "CREATE VIEW active_users",
        "DROP TABLE legacy_a",
        "DROP TABLE legacy_b",
        "CREATE FUNCTION touch",
        "CREATE TRIGGER trg_touch",
        "CREATE PROCEDURE cleanup",
        "INSERT users",
    ]
    assert migration.meta["operation_count"] == 19
    assert migration.signature.startswith("MIGRATION 2024_01_15_120000 create_core [up]: ")
    assert len(migration.signature) <= MAX_SIGNATURE_CHARS
    index = _index(_write(tmp_path / "conv", {
        "sql/V2_1__add_coupons.sql": "CREATE TABLE coupons (id INT);\n",
        "sql/U2_1__add_coupons.sql": "DROP TABLE coupons;\n",
        "db/migrate/003_drop_legacy.down.sql": "CREATE TABLE legacy (id INT);\n",
        "db/migrate/003_drop_legacy.up.sql": "DROP TABLE legacy;\n",
        "migrations/seed.sql": "INSERT INTO coupons (id) VALUES (1);\n",
        "sql/0007-add-index.sql": "CREATE INDEX i ON coupons (id);\n",
        "db/schema.sql": "CREATE TABLE plain (id INT);\n",
        "goose/20240102030405_both.sql": (
            "-- +goose Up\nCREATE TABLE g (id INT);\n-- +goose Down\nDROP TABLE g;\n"
        ),
    }))
    found = {s.file: s for s in index.iter_symbols("migration")}
    assert "db/schema.sql" not in found  # plain schema file: not a migration
    flyway = found["sql/V2_1__add_coupons.sql"]
    assert (flyway.meta["version"], flyway.meta["direction"], flyway.meta["style"]) == (
        "2.1", "up", "flyway",
    )
    assert flyway.meta["operations"] == ["CREATE TABLE coupons"]
    assert found["sql/U2_1__add_coupons.sql"].meta["direction"] == "down"
    down = found["db/migrate/003_drop_legacy.down.sql"]
    assert (down.meta["version"], down.meta["direction"], down.name) == (
        "003", "down", "drop_legacy",
    )
    assert found["db/migrate/003_drop_legacy.up.sql"].meta["operations"] == ["DROP TABLE legacy"]
    assert found["migrations/seed.sql"].meta["style"] == "named"  # by directory convention
    assert found["sql/0007-add-index.sql"].meta["version"] == "0007"  # numeric prefix
    both = found["goose/20240102030405_both.sql"]
    assert both.meta["direction"] == "up+down"
    assert both.meta["operations"] == ["CREATE TABLE g"]
    assert both.meta["down_operations"] == ["DROP TABLE g"]


# ======================================================================================
# Gate 5: Walker Gate
# ======================================================================================


def test_walker_prunes_ignored_dirs_at_any_depth(tmp_path: Path, monkeypatch: Any) -> None:
    """.git / .harness / .venv / node_modules / vendor are never descended, at any depth."""
    assert DEFAULT_IGNORE_DIRS == {".git", ".harness", ".venv", "node_modules", "vendor"}
    _write(tmp_path, {
        "keep.py": "def keep(): pass\n",
        "a/b/c/deep.py": "def deep(): pass\n",
        ".git/hooks/hook.py": "def bad(): pass\n",
        ".harness/state.py": "def bad(): pass\n",
        ".venv/lib/site.py": "def bad(): pass\n",
        "node_modules/pkg/index.js": "export const bad = 1;\n",
        "vendor/lib.php": "<?php function bad() {}\n",
        "a/b/node_modules/x/y/z.js": "export const bad = 1;\n",
        "a/b/c/vendor/v.php": "<?php function bad() {}\n",
        "a/.git/objects/o.py": "def bad(): pass\n",
        "x/y/z/.venv/Lib/site-packages/p.py": "def bad(): pass\n",
        "x/y/.harness/cache/c.sql": "CREATE TABLE bad (id INT);\n",
        "a/vendor_not/ok.php": "<?php function ok() {}\n",  # only exact names are ignored
    })
    visited: list[str] = []
    real_scandir = os.scandir

    def spy(path: Any) -> Any:
        visited.append(str(path))
        return real_scandir(path)

    monkeypatch.setattr(scout.os, "scandir", spy)
    index = _index(tmp_path)
    assert [rec.path for rec in index.files] == ["a/b/c/deep.py", "a/vendor_not/ok.php", "keep.py"]
    assert not index.find_symbol("bad")
    assert index.stats["skipped"]["ignored_dirs"] == 10
    for scanned in visited:
        parts = set(Path(scanned).relative_to(tmp_path).parts)
        assert not parts & DEFAULT_IGNORE_DIRS, f"descended into {scanned}"


def test_walker_extra_ignore_names_globs_and_virtualenvs(tmp_path: Path) -> None:
    _write(tmp_path, {
        "src/app.py": "x = 1\n",
        "src/dist/bundle.js": "export const a = 1;\n",
        "dist/out.js": "export const a = 1;\n",
        "src/schema.generated.ts": "export const a = 1;\n",
        "docs/conf.py": "x = 1\n",
        "src/docs.py": "x = 1\n",
        "env/pyvenv.cfg": "home = /usr\n",
        "env/lib/thing.py": "x = 1\n",
        "build/gen/deep.py": "x = 1\n",
    })
    default = [rec.path for rec in _index(tmp_path).files]
    assert "env/lib/thing.py" not in default and "dist/out.js" in default
    custom = _index(tmp_path, extra_ignore=["dist", "*.generated.ts", "docs/", "build/gen"])
    assert [rec.path for rec in custom.files] == ["src/app.py", "src/docs.py"]
    assert custom.stats["skipped"]["ignored_files"] == 1
    # a single string is accepted, replacing ignore_dirs is possible, venv pruning can be off
    assert "dist/out.js" not in [r.path for r in _index(tmp_path, extra_ignore="dist").files]
    opened = _index(tmp_path, skip_virtualenvs=False, ignore_dirs=["src"])
    assert [rec.path for rec in opened.files] == [
        "build/gen/deep.py", "dist/out.js", "docs/conf.py", "env/lib/thing.py",
    ]
    only_sql = _index(_write(tmp_path, {"t.sql": "CREATE TABLE t (id INT);\n"}), languages={"sql"})
    assert [rec.path for rec in only_sql.files] == ["t.sql"]


def test_walker_does_not_follow_symlinked_dirs_and_is_loop_safe(tmp_path: Path) -> None:
    repo = _write(tmp_path / "repo", {"pkg/mod.py": "def f(): pass\n"})
    outside = _write(tmp_path / "outside", {"secret.py": "def leaked(): pass\n"})
    _link_dir(repo / "linked", outside)
    _link_dir(repo / "pkg" / "loop", repo)  # a cycle back to the root
    default = _index(repo)
    assert [rec.path for rec in default.files] == ["pkg/mod.py"]
    assert default.stats["skipped"]["symlinks"] == 2
    followed = _index(repo, follow_symlinks=True)  # terminates despite the cycle
    assert [rec.path for rec in followed.files] == ["linked/secret.py", "pkg/mod.py"]


def test_walker_skips_binary_oversize_and_minified(tmp_path: Path) -> None:
    minified = "var a=1;" * 2000
    _write(tmp_path, {
        "ok.py": "def ok(): pass\n",
        "blob.py": b"\x00\x01\x02binary\x00data",
        "big.py": "def big(): pass\n" + "# padding\n" * 400,
        "lib/app.min.js": "function m(){}\n",
        "lib/bundle.js": minified,
        "lib/real.js": "function real() {}\n",
    })
    index = _index(tmp_path, max_file_bytes=2000)
    by_path = {rec.path: rec for rec in index.files}
    assert by_path["ok.py"].extractor == scout.EXTRACTOR_PY_AST
    for path, category in (("blob.py", "binary"), ("big.py", "too_large"),
                           ("lib/app.min.js", "minified")):
        rec = by_path[path]
        assert rec.extractor == scout.EXTRACTOR_SKIPPED and rec.symbols == () and rec.edges == ()
        assert rec.errors[0].startswith(f"skipped [{category}]"), rec.errors
    assert by_path["lib/bundle.js"].extractor == scout.EXTRACTOR_SKIPPED  # also over the cap
    assert by_path["big.py"].size_bytes > 2000 and len(by_path["big.py"].sha256) == 64
    assert index.stats["skipped"]["binary"] == 1 and index.stats["skipped"]["minified"] == 1
    assert index.stats["skipped"]["too_large"] == 2
    assert _names(by_path["lib/real.js"], "function") == ["real"]
    # the minified heuristic also catches unnamed bundles, and it is configurable
    relaxed = _index(tmp_path, max_file_bytes=10_000_000)
    assert relaxed.file("lib/bundle.js").errors[0].startswith("skipped [minified]")
    assert relaxed.file("big.py").extractor == scout.EXTRACTOR_PY_AST
    kept = _index(tmp_path, max_file_bytes=10_000_000, skip_minified=False)
    assert kept.file("lib/bundle.js").extractor == "javascript-regex"
    assert _names(kept.file("lib/app.min.js"), "function") == ["m"]


def test_walker_encodings_crlf_empty_and_unreadable(tmp_path: Path, monkeypatch: Any) -> None:
    _write(tmp_path, {
        "bom.py": b"\xef\xbb\xbfdef with_bom(): pass\n",
        "utf16.py": "def wide(): pass\n".encode("utf-16"),
        "latin.py": b"# caf\xe9\ndef latin(): pass\n",
        "crlf.php": b"<?php\r\nfunction one() {}\r\n\r\nfunction two() {}\r\n",
        "old_mac.js": b"function a() {}\rfunction b() {}\r",
        "empty.ts": b"",
        "locked.py": b"def locked(): pass\n",
        "fine.sql": b"CREATE TABLE t (id INT);\n",
    })
    real_getsize = os.path.getsize

    def getsize(path: Any) -> int:
        if str(path).endswith("locked.py"):
            raise PermissionError(13, "Permission denied")
        return real_getsize(path)

    monkeypatch.setattr(scout.os.path, "getsize", getsize)
    index = _index(tmp_path)
    by_path = {rec.path: rec for rec in index.files}
    assert len(by_path) == 8  # every file is accounted for; nothing raised
    assert _names(by_path["bom.py"], "function") == ["with_bom"] and not by_path["bom.py"].errors
    assert _names(by_path["utf16.py"], "function") == ["wide"]
    assert _names(by_path["latin.py"], "function") == ["latin"]
    assert by_path["latin.py"].errors == ("encoding: not valid UTF-8; decoded as latin-1",)
    crlf = by_path["crlf.php"]
    assert [(s.name, s.line) for s in crlf.symbols] == [("one", 2), ("two", 4)]
    assert crlf.line_count == 4
    assert [(s.name, s.line) for s in by_path["old_mac.js"].symbols] == [("a", 1), ("b", 2)]
    empty = by_path["empty.ts"]
    assert (empty.size_bytes, empty.line_count, empty.symbols, empty.errors) == (0, 0, (), ())
    assert empty.extractor == "typescript-regex"
    locked = by_path["locked.py"]
    assert locked.extractor == scout.EXTRACTOR_SKIPPED
    assert locked.errors[0].startswith("skipped [unreadable]: PermissionError")
    assert index.stats["skipped"]["unreadable"] == 1 and index.stats["files_with_errors"] == 2
    assert _names(by_path["fine.sql"], "table") == ["t"]


def test_paths_are_repo_relative_posix(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    for rec in index.files:
        assert "\\" not in rec.path and not rec.path.startswith("/") and ":" not in rec.path
        assert all(s.file == rec.path for s in rec.symbols)
        assert all(e.file == rec.path for e in rec.edges)
        assert all("\\" not in (e.resolved_file or "") for e in rec.edges)
    assert "web/src/components/UserCard.tsx" in {rec.path for rec in index.files}
    assert "\\" not in index.root
    # lookups accept OS-style separators and absolute paths
    assert index.file("web\\src\\index.js") is index.file("web/src/index.js")
    indexer = _indexer(tmp_path)
    assert indexer.index_file(tmp_path / "svc" / "app" / "db.py").path == "svc/app/db.py"
    assert indexer.index_file("svc\\app\\db.py").path == "svc/app/db.py"


def test_output_is_deterministic_and_byte_identical(tmp_path: Path) -> None:
    """Two runs (serial or threaded, same or copied tree) serialize to identical JSON."""
    repo = _write(tmp_path / "one", POLYREPO)
    first = _index(repo)
    second = _index(repo)
    assert first.to_json() == second.to_json()
    assert first.to_json(indent=2) == second.to_json(indent=2)
    threaded = _index(repo, workers=8)
    assert threaded.to_json() == first.to_json()
    assert [rec.path for rec in first.files] == sorted(rec.path for rec in first.files)
    assert first.render_outline(2000) == threaded.render_outline(2000)
    assert json.dumps(first.to_summary(2000)) == json.dumps(threaded.to_summary(2000))
    # written in reverse order into another directory: same digest, only ``root`` differs
    clone = tmp_path / "two"
    _write(clone, dict(reversed(list(POLYREPO.items()))))
    other = _index(clone)
    assert other.digest == first.digest and other.root != first.root
    payload, other_payload = first.to_dict(), other.to_dict()
    payload.pop("root")
    other_payload.pop("root")
    assert payload == other_payload
    assert json.loads(first.to_json())["stats"] == first.stats
    assert "elapsed" not in first.to_json()  # nothing run-specific is serialized


# ======================================================================================
# Gate 6: Data Model Gate
# ======================================================================================


def test_index_round_trip_and_schema_version(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    payload = index.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION == index.schema_version
    assert payload["digest"] == index.digest and len(index.digest) == 64
    assert set(payload) == {"schema_version", "root", "digest", "stats", "files"}
    restored = RepoIndex.from_json(index.to_json())
    assert restored.to_json() == index.to_json()
    assert restored.files == index.files and restored.digest == index.digest
    assert RepoIndex.from_dict(json.loads(index.to_json(indent=1))).files == index.files
    # per-record round trips keep every field
    rec = index.file("src/Http/UserController.php")
    assert rec is not None and FileRecord.from_dict(rec.to_dict()) == rec
    sym = rec.symbols[0]
    assert Symbol.from_dict(sym.to_dict()) == sym
    edge = next(e for e in rec.edges if e.resolved_file)
    assert Edge.from_dict(edge.to_dict()) == edge
    # the digest tracks content, not location
    (tmp_path / "svc/app/db.py").write_text("def get_conn():\n    return 1\n", encoding="utf-8")
    assert _index(tmp_path).digest != index.digest


def test_index_format_errors() -> None:
    with pytest.raises(IndexFormatError):
        RepoIndex.from_json("{not json")
    with pytest.raises(IndexFormatError, match="schema_version"):
        RepoIndex.from_dict({"schema_version": SCHEMA_VERSION + 1, "files": []})
    with pytest.raises(IndexFormatError):
        RepoIndex.from_dict({"schema_version": SCHEMA_VERSION, "files": [{"path": "x.py"}]})
    with pytest.raises(IndexFormatError):
        RepoIndex.from_dict([])  # type: ignore[arg-type]
    assert issubclass(IndexFormatError, ScoutError) and issubclass(IndexFormatError, ValueError)
    empty = RepoIndex.from_dict({"schema_version": SCHEMA_VERSION})
    assert empty.files == () and empty.render_outline(100).startswith("# repo outline: 0 files")


def test_query_helpers(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    assert len(list(index.iter_symbols())) == index.stats["symbols"]
    assert len(list(index.iter_edges())) == index.stats["edges"]
    assert {s.language for s in index.iter_symbols(language="sql")} == {"sql"}
    assert {s.qualified_name for s in index.iter_symbols("table")} == {"users", "orders", "coupons"}
    assert {e.kind for e in index.iter_edges("foreign_key")} == {"foreign_key"}
    (user,) = index.find_symbol("App\\Models\\User")
    assert user.kind == "class" and user.file == "src/Models/User.php"
    assert {s.file for s in index.find_symbol("User")} == {
        "src/Models/User.php", "web/src/types/index.ts",
    }
    assert [s.kind for s in index.find_symbol("User", kind="interface")] == ["interface"]
    assert index.find_symbol("User", language="php")[0].qualified_name == "App\\Models\\User"
    assert index.find_symbol("User", path="web/src/types/index.ts")[0].kind == "interface"
    assert index.find_symbol("NoSuchThing") == []
    assert [s.name for s in index.symbols_in("svc/app/models.py")] == ["Account", "id"]
    assert index.symbols_in("missing.py") == ()
    assert index.file("missing.py") is None
    assert index.dependencies_of("svc/app/main.py") == [
        "db/schema.sql", "svc/app/db.py", "svc/app/models.py",
    ]
    assert index.dependents_of("svc/app/models.py") == ["svc/app/main.py"]
    assert index.dependents_of("src/Models/User.php") == ["src/Http/UserController.php"]
    assert "web/src/api/client.ts" in index.dependents_of("routes/api.php")
    assert index.dependencies_of("missing.py") == [] and index.dependents_of("missing.py") == []
    stats = index.stats
    assert stats["files"] == len(index.files) == 20
    assert stats["by_language"]["php"]["files"] == 8
    assert stats["resolved_edges"] == sum(1 for e in index.iter_edges() if e.resolved_file)
    assert stats["skipped"]["ignored_dirs"] == 2  # vendor/ and web/node_modules/


def test_index_never_carries_file_bodies(tmp_path: Path) -> None:
    """A large file yields a small index and no captured string exceeds the documented cap."""
    marker = "BODY_MARKER_7f3a"
    params = ", ".join(f"parameter_number_{i}: int = {i}" for i in range(120))
    py_lines = [
        '"""' + "module doc " * 60 + '"""',
        f"def wide({params}) -> None:",
        '    """' + "doc " * 200 + '"""',
        "    pass",
        "BIG = " + repr("x" * 5000),
        "SQL = " + repr("SELECT " + ", ".join(f"c{i}" for i in range(500)) + " FROM t"),
    ]
    for i in range(60):
        py_lines.append(f"def fn_{i}(a, b):")
        py_lines.extend(f"    v{j} = '{marker} {i} {j} ' * 3" for j in range(150))
        py_lines.append("    return cursor.execute(SQL)")
    php_lines = ["<?php", "class Big {"]
    for i in range(50):
        php_lines.append(f"    public function m{i}(array $a): array {{")
        php_lines.extend(f"        $v{j} = '{marker} {i} {j}';" for j in range(120))
        php_lines.append("        return $a;\n    }")
    php_lines.append("}")
    js_lines = [f"const LONG = '{'y' * 4000}';"]
    for i in range(50):
        js_lines.append(f"export function f{i}(a, b) {{")
        js_lines.extend(f"  const v{j} = `{marker} {i} {j} ${{a}}`;" for j in range(120))
        js_lines.append("  return fetch('/api/thing/' + a);\n}")
    sql_lines = ["CREATE TABLE wide ("]
    sql_lines.append(",\n".join(f"  column_{i} VARCHAR(255) DEFAULT '{marker}'" for i in range(400)))
    sql_lines.append(");")
    sql_lines.extend(f"INSERT INTO wide (column_0) VALUES ('{marker} {i}');" for i in range(500))
    files = {
        "big.py": "\n".join(py_lines) + "\n",
        "Big.php": "\n".join(php_lines) + "\n",
        "big.js": "\n".join(js_lines) + "\n",
        "big.sql": "\n".join(sql_lines) + "\n",
    }
    index = _index(_write(tmp_path, files))
    assert all(not rec.errors for rec in index.files), [rec.errors for rec in index.files]
    blob = index.to_json()
    source_bytes = sum(len(text) for text in files.values())
    assert source_bytes > 800_000
    assert len(blob) < source_bytes / 10  # signatures and edges only: a fraction of the source
    assert marker not in blob
    assert marker not in index.render_outline(1_000_000)
    assert marker not in json.dumps(index.to_summary(1_000_000))
    assert MAX_CAPTURED_CHARS >= max(MAX_SIGNATURE_CHARS, MAX_DOC_CHARS, MAX_SQL_PREVIEW_CHARS)
    longest = max(_strings(index.to_dict()["files"]), key=len)
    assert len(longest) <= MAX_CAPTURED_CHARS, longest
    for sym in index.iter_symbols():
        assert len(sym.signature) <= MAX_SIGNATURE_CHARS
        assert sym.doc is None or len(sym.doc) <= MAX_DOC_CHARS
    wide = index.find_symbol("wide", kind="function")[0]
    assert wide.signature.startswith("def wide(parameter_number_0: int = 0, ")
    assert wide.signature.endswith("...") and len(wide.signature) == MAX_SIGNATURE_CHARS
    assert index.find_symbol("BIG")[0].signature == "BIG = ..."
    table = index.find_symbol("wide", kind="table")[0]
    assert table.meta["column_count"] == 400 and len(table.meta["columns"]) == scout.MAX_META_ITEMS
    # per-file symbol / edge caps are enforced and reported
    capped = _index(tmp_path, max_symbols_per_file=50, max_edges_per_file=40)
    big = capped.file("big.py")
    assert big is not None and len(big.symbols) == 50 and len(big.edges) == 40
    assert any(e.startswith("truncated:") and "symbols omitted" in e for e in big.errors)
    assert any(e.startswith("truncated:") and "edges omitted" in e for e in big.errors)


# ======================================================================================
# Gate 7: Incremental Reuse Gate
# ======================================================================================


def test_incremental_index_reuses_unchanged_files(tmp_path: Path, monkeypatch: Any) -> None:
    repo = _write(tmp_path, POLYREPO)
    indexer = _indexer(repo)
    first = indexer.index()
    assert indexer.last_run.files_total == 20 and indexer.last_run.files_parsed == 20
    assert indexer.last_run.files_reused == 0

    parsed: list[str] = []
    real_index_bytes = PolyglotRepoIndexer._index_bytes

    def counting(self: PolyglotRepoIndexer, rel: str, *args: Any) -> FileRecord:
        parsed.append(rel)
        return real_index_bytes(self, rel, *args)

    monkeypatch.setattr(PolyglotRepoIndexer, "_index_bytes", counting)
    second = indexer.index(previous=first)
    assert parsed == [] and indexer.last_run.files_reused == 20
    assert second.to_json() == first.to_json()
    assert all(a is b for a, b in zip(first.files, second.files, strict=True))

    (repo / "svc/app/db.py").write_text(
        "def get_conn():\n    return 1\n\n\ndef extra():\n    return 2\n", encoding="utf-8"
    )
    third = indexer.index(previous=second)
    assert parsed == ["svc/app/db.py"]
    assert (indexer.last_run.files_parsed, indexer.last_run.files_reused) == (1, 19)
    assert third.find_symbol("extra") and third.digest != second.digest
    assert third.file("svc/app/models.py") is second.file("svc/app/models.py")
    assert third.to_json() == _index(repo).to_json()  # identical to a cold run

    # reuse also works from a deserialized index, and across a new indexer instance
    parsed.clear()
    reloaded = RepoIndex.from_json(third.to_json())
    fourth = _indexer(repo).index(previous=reloaded)
    assert parsed == [] and fourth.to_json() == third.to_json()


def test_incremental_index_re_resolves_edges_and_handles_removed_files(tmp_path: Path) -> None:
    repo = _write(tmp_path / "repo", POLYREPO)
    indexer = _indexer(repo)
    first = indexer.index()
    main = first.file("svc/app/main.py")
    assert main is not None
    assert _edge(main, "import", "svc.app.models.Account").resolved_file == "svc/app/models.py"
    (repo / "svc/app/models.py").unlink()
    (repo / "svc/app/extra.py").write_text("def added(): pass\n", encoding="utf-8")
    second = indexer.index(previous=first)
    assert (indexer.last_run.files_parsed, indexer.last_run.files_reused) == (1, 19)
    assert second.file("svc/app/models.py") is None and second.find_symbol("added")
    main = second.file("svc/app/main.py")
    assert main is not None
    stale = _edge(main, "import", "svc.app.models.Account")
    assert stale.resolved_file is None and stale.resolved_symbol is None
    assert _edge(main, "extends", "Account").resolved_file is None
    assert _edge(main, "import", "svc.app.db.get_conn").resolved_file == "svc/app/db.py"
    assert second.to_json() == _index(repo).to_json()
    # skipped / failed records are never reused; a foreign previous index is harmless
    other = _index(_write(tmp_path / "other", {"svc/app/db.py": "def different(): pass\n"}))
    third = indexer.index(previous=other)
    assert third.to_json() == second.to_json() and indexer.last_run.files_reused == 0


# ======================================================================================
# Gate 8: Cross-File Resolution Gate
# ======================================================================================


def test_resolution_python_imports(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    main = index.file("svc/app/main.py")
    assert main is not None
    resolved = {e.target: (e.resolved_file, e.resolved_symbol) for e in _edges(main, "import")}
    assert resolved == {
        "fastapi.FastAPI": (None, None),  # third party: never guessed
        "svc.app.db.get_conn": ("svc/app/db.py", "get_conn"),  # relative, to a symbol
        "svc.app.models": ("svc/app/models.py", None),  # ``from . import module``
        "svc.app.models.Account": ("svc/app/models.py", "Account"),
        "svc.app.db": ("svc/app/db.py", None),  # absolute dotted import
    }
    base = _edge(main, "extends", "Account")  # through the import table of the file
    assert (base.resolved_file, base.resolved_symbol) == ("svc/app/models.py", "Account")
    pkg = _index(_write(tmp_path / "pkg", {
        "lib/__init__.py": "from .core import run\n",
        "lib/core.py": "def run(): pass\n",
        "app.py": "import lib\nfrom lib import core\nfrom lib.missing import nope\n",
        "one/util.py": "x = 1\n",
        "two/util.py": "x = 2\n",
        "ambiguous.py": "import util\n",
    }))
    app = pkg.file("app.py")
    assert app is not None
    assert _edge(app, "import", "lib").resolved_file == "lib/__init__.py"  # package
    assert _edge(app, "import", "lib.core").resolved_file == "lib/core.py"
    assert _edge(app, "import", "lib.missing.nope").resolved_file is None
    init = pkg.file("lib/__init__.py")
    assert init is not None and _edge(init, "import", "lib.core.run").resolved_symbol == "run"
    ambiguous = pkg.file("ambiguous.py")
    assert ambiguous is not None
    assert _edge(ambiguous, "import", "util").resolved_file is None  # two candidates: no guess
    assert "resolution" not in _edge(app, "import", "lib.core").meta  # exact match: no label
    # a src-layout module is found by its unique dotted-name suffix, and labelled as inferred
    src = _index(_write(tmp_path / "src_layout", {
        "src/shop/cart.py": "def total(): pass\n",
        "tools/run.py": "from shop.cart import total\nimport shop.cart\n",
    }))
    run = src.file("tools/run.py")
    assert run is not None
    inferred = _edge(run, "import", "shop.cart.total")
    assert (inferred.resolved_file, inferred.resolved_symbol) == ("src/shop/cart.py", "total")
    assert inferred.meta["resolution"] == "suffix"
    assert _edge(run, "import", "shop.cart").meta == {"resolution": "suffix"}


def test_resolution_php_symbol_table_and_psr4(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    user = index.file("src/Models/User.php")
    controller = index.file("src/Http/UserController.php")
    assert user is not None and controller is not None
    ext = _edge(user, "extends", "App\\Models\\Model")
    assert (ext.resolved_file, ext.resolved_symbol) == ("src/Models/Model.php", "App\\Models\\Model")
    impl = _edge(user, "implements", "App\\Contracts\\HasName")
    assert impl.resolved_file == "src/Contracts/HasName.php"
    trait = _edge(user, "uses_trait", "App\\Support\\Timestamps")
    assert trait.resolved_file == "src/Support/Timestamps.php"
    use = _edge(controller, "import", "Lib\\Mailer")
    assert (use.resolved_file, use.resolved_symbol) == ("lib/Mailer.php", "Lib\\Mailer")
    fn = _edge(controller, "import", "App\\Support\\slugify")
    assert (fn.resolved_file, fn.resolved_symbol) == (
        "src/Support/helpers.php", "App\\Support\\slugify",
    )
    new = [e for e in _edges(controller, "call") if e.meta.get("via") == "new"]
    assert {(e.target, e.resolved_file) for e in new} == {
        ("Lib\\Mailer", "lib/Mailer.php"),
        ("App\\Models\\User", "src/Models/User.php"),
        ("App\\Jobs\\SendWelcome", None),  # no such class anywhere: unresolved
    }
    static = _edge(controller, "call", "App\\Models\\User::find")
    assert static.resolved_symbol == "App\\Models\\User::find"
    assert _edge(controller, "import", "App\\Jobs\\SendWelcome").resolved_file is None
    # composer.json PSR-4 alone locates a class, and breaks ties between duplicate declarations
    psr = _index(_write(tmp_path / "psr", {
        "composer.json": json.dumps({"autoload": {"psr-4": {"Acme\\": "src"}},
                                     "autoload-dev": {"psr-4": {"Acme\\Tests\\": "tests/"}}}),
        "src/Billing/Invoice.php": "<?php\nnamespace Acme\\Billing;\nclass Invoice {}\n",
        "legacy/Invoice.php": "<?php\nnamespace Acme\\Billing;\nclass Invoice {}\n",
        "src/Billing/Opaque.php": "<?php\n// generated at runtime\n",
        "tests/InvoiceTest.php": (
            "<?php\nnamespace Acme\\Tests;\nuse Acme\\Billing\\Invoice;\nuse Acme\\Billing\\Opaque;\n"
            "class InvoiceTest { function t() { return new Invoice(); } }\n"
        ),
        "dup/a.php": "<?php\nnamespace Dup;\nclass Twice {}\n",
        "dup/b.php": "<?php\nnamespace Dup;\nclass Twice {}\n",
        "dup/use.php": "<?php\nuse Dup\\Twice;\n",
    }))
    test_file = psr.file("tests/InvoiceTest.php")
    assert test_file is not None
    assert _edge(test_file, "import", "Acme\\Billing\\Invoice").resolved_file == (
        "src/Billing/Invoice.php"
    )
    opaque = _edge(test_file, "import", "Acme\\Billing\\Opaque")
    assert (opaque.resolved_file, opaque.resolved_symbol) == ("src/Billing/Opaque.php", None)
    dup = psr.file("dup/use.php")
    assert dup is not None and _edge(dup, "import", "Dup\\Twice").resolved_file is None


def test_resolution_js_relative_specifiers(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    card = index.file("web/src/components/UserCard.tsx")
    entry = index.file("web/src/index.js")
    assert card is not None and entry is not None
    resolved = {e.target: e.resolved_file for e in _edges(card, "import")}
    assert resolved == {
        "react": None,  # bare package specifier
        "../api/client": "web/src/api/client.ts",  # extension resolution
        "./BaseCard": "web/src/components/BaseCard.jsx",
        "../types": "web/src/types/index.ts",  # index-file resolution
        "./does-not-exist": None,
    }
    base = _edge(card, "extends", "BaseCard")
    assert (base.resolved_file, base.resolved_symbol) == (
        "web/src/components/BaseCard.jsx", "BaseCard",
    )
    assert _edge(card, "implements", "Renderable").resolved_file is None
    assert _edge(entry, "import", "./components/UserCard").resolved_file == (
        "web/src/components/UserCard.tsx"
    )
    assert _edge(entry, "import", "./util").resolved_file == "web/src/util/index.js"  # require
    # TS-style ``./client.js`` specifier that really points at client.ts
    assert _edge(entry, "export", "*").resolved_file == "web/src/api/client.ts"
    escaping = _index(_write(tmp_path / "esc", {"a.js": "import x from '../../outside';\n"}))
    assert _edges(escaping.files[0], "import")[0].resolved_file is None


def test_resolution_sql_foreign_keys_and_embedded_sql(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    schema = index.file("db/schema.sql")
    migration = index.file("db/migrations/002_add_coupons.sql")
    controller = index.file("src/Http/UserController.php")
    main = index.file("svc/app/main.py")
    assert schema and migration and controller and main
    (fk,) = _edges(schema, "foreign_key")
    assert (fk.source, fk.target, fk.resolved_file, fk.resolved_symbol) == (
        "orders", "users", "db/schema.sql", "users",
    )
    fks = {e.target: e for e in _edges(migration, "foreign_key")}
    assert fks["coupons"].resolved_file == "db/migrations/002_add_coupons.sql"
    assert fks["regions"].resolved_file is None  # table defined nowhere
    alter = [e for e in _edges(migration, "sql_table") if e.meta["verb"] == "ALTER"]
    assert {e.resolved_file for e in alter} == {"db/schema.sql"}  # to the defining file
    # SQL found in PHP and in Python resolves to the table definition
    php_sql = {(e.target, e.meta["verb"]): e.resolved_file for e in _edges(controller, "sql_table")}
    assert php_sql == {
        ("users", "SELECT"): "db/schema.sql",
        ("orders", "INSERT"): "db/schema.sql",
        ("ghosts", "DELETE"): None,
    }
    (py_sql,) = _edges(main, "sql_table")
    assert (py_sql.source, py_sql.target, py_sql.resolved_file, py_sql.resolved_symbol) == (
        "health", "users", "db/schema.sql", "users",
    )
    dup = _index(_write(tmp_path / "dup", {
        "a.sql": "CREATE TABLE things (id INT);\n",
        "b.sql": "CREATE TABLE things (id INT);\n",
        "q.php": "<?php $pdo->query('SELECT * FROM things');\n",
    }))
    query = dup.file("q.php")
    assert query is not None and _edges(query, "sql_table")[0].resolved_file is None  # ambiguous


def test_resolution_api_calls_to_server_routes(tmp_path: Path) -> None:
    """fetch / axios URLs match PHP and Python routes with path-parameter awareness."""
    index = _index(_write(tmp_path, POLYREPO))
    client = index.file("web/src/api/client.ts")
    assert client is not None
    resolved = {
        e.target: (e.resolved_file, e.resolved_symbol) for e in _edges(client, "api_call")
    }
    assert resolved == {
        "GET /api/users/{id}": ("routes/api.php", "GET /api/users/{id}"),  # `${BASE}/users/${id}`
        "GET /api/users": ("routes/api.php", "GET /api/users"),
        "POST /api/users": ("routes/api.php", "POST /api/users"),
        "DELETE /api/users/{id}": ("routes/api.php", "DELETE /api/users/{id}"),
        "GET /py/health": ("svc/app/main.py", "GET /py/health"),  # Python route
        "GET /py/accounts/{id}": ("svc/app/main.py", "GET /py/accounts/{account_id}"),
        "GET /nowhere/at/all": (None, None),
    }
    # route symbols link to their controller methods
    routes = index.file("routes/api.php")
    assert routes is not None
    handler = next(e for e in _edges(routes, "route") if e.source == "GET /api/users/{id}")
    assert (handler.resolved_file, handler.resolved_symbol) == (
        "src/Http/UserController.php", "App\\Http\\UserController::show",
    )
    missing = next(e for e in _edges(routes, "route") if e.source == "GET /api/orders")
    assert missing.resolved_file is None
    # the HTTP method takes part in matching; ambiguity is never guessed
    edge_cases = _index(_write(tmp_path / "edge", {
        "routes.php": (
            "<?php\nRoute::get('/things/{id}', 'ThingController@show');\n"
            "Route::get('/dup', 'A@one');\n"
        ),
        "more.php": "<?php\nRoute::get('/dup', 'B@two');\n",
        "app/Http/Controllers/ThingController.php": (
            "<?php\nnamespace App\\Http\\Controllers;\n"
            "class ThingController { public function show($id) {} }\n"
        ),
        "app.js": (
            "fetch('/things/42');\n"
            "fetch('/things/42', { method: 'DELETE' });\n"
            "fetch('/dup');\n"
            "fetch('https://api.example.com/things/7');\n"
            "fetch('/backend/v2/things/9');\n"
        ),
    }))
    app = edge_cases.file("app.js")
    assert app is not None
    by_target = {e.target: e for e in _edges(app, "api_call")}
    assert {target: e.resolved_symbol for target, e in by_target.items()} == {
        "GET /things/42": "GET /things/{id}",
        "DELETE /things/42": None,  # no route for that verb
        "GET /dup": None,  # two candidate routes
        "GET https://api.example.com/things/7": "GET /things/{id}",  # origin stripped
        "GET /backend/v2/things/9": "GET /things/{id}",  # unknown mount prefix: suffix match
    }
    # an inferred match is never silent: it is labelled on the edge and in the summary
    assert by_target["GET /backend/v2/things/9"].meta["resolution"] == "suffix"
    assert by_target["GET https://api.example.com/things/7"].meta["resolution"] == "origin-ignored"
    assert "resolution" not in by_target["GET /things/42"].meta
    assert "resolution" not in by_target["GET /dup"].meta
    calls = {c["call"]: c for c in edge_cases.to_summary(5000)["api_calls"]}
    assert calls["GET /backend/v2/things/9"]["match"] == "suffix"
    assert "match" not in calls["GET /things/42"]
    php_routes = edge_cases.file("routes.php")
    assert php_routes is not None
    short = next(e for e in _edges(php_routes, "route") if e.source == "GET /things/{id}")
    assert (short.target, short.resolved_file, short.resolved_symbol) == (
        "ThingController::show", "app/Http/Controllers/ThingController.php",
        "App\\Http\\Controllers\\ThingController::show",
    )
    assert short.meta["resolution"] == "short-name"  # 'ThingController@show' has no namespace
    # labels are recomputed, not sticky: once the ambiguity is gone the edge resolves exactly
    (tmp_path / "edge" / "more.php").unlink()
    again = _indexer(tmp_path / "edge").index(previous=edge_cases).file("app.js")
    assert again is not None and _edge(again, "api_call", "GET /dup").resolved_symbol == "GET /dup"


def test_unresolved_edges_keep_none() -> None:
    assert Edge("call", "a", "b", "f.py", 1).resolved_file is None
    assert "resolved_file" not in Edge("call", "a", "b", "f.py", 1).to_dict()


# ======================================================================================
# Gate 9: Downstream Hooks Gate (render_outline / to_summary)
# ======================================================================================


def _synthetic_index(n_files: int) -> RepoIndex:
    """A large in-memory index: PHP controllers, TS clients, SQL tables and Python modules."""
    records: list[FileRecord] = []
    for i in range(n_files):
        lang, ext = (("php", "php"), ("typescript", "ts"), ("sql", "sql"), ("python", "py"))[i % 4]
        path = f"mod{i % 40:02d}/pkg{i % 7}/file_{i:05d}.{ext}"
        symbols: list[Symbol] = []
        edges: list[Edge] = []
        if lang == "sql":
            symbols.append(Symbol(
                "table", f"t{i}", f"t{i}", f"TABLE t{i} (id INT PK, name TEXT)", path, 1, 4, lang,
                meta={"columns": ["id INT PK", "name TEXT"], "primary_key": ["id"]},
            ))
            edges.append(Edge("foreign_key", f"t{i}", "t2", path, 3, "mod02/pkg2/file_00002.sql",
                              "t2", {"columns": ["ref"], "ref_columns": ["id"]}))
        else:
            cls = f"Thing{i}"
            symbols.append(Symbol("class", cls, cls, f"class {cls} extends Base{i % 9}", path, 3,
                                  90, lang, doc=f"Thing number {i}."))
            for j in range(8):
                symbols.append(Symbol(
                    "method", f"m{j}", f"{cls}.m{j}",
                    f"public function m{j}(int $a, string $b = 'x'): array", path, 5 + j * 10,
                    12 + j * 10, lang, parent=cls,
                ))
            symbols.append(Symbol("constant", "LIMIT", f"{cls}.LIMIT", "const LIMIT = 5", path, 4,
                                  4, lang, parent=cls))
            label = f"GET /api/things{i}/{{id}}"
            if lang == "php":
                symbols.append(Symbol(
                    "route", label, label, f"ROUTE {label} -> {cls}::m0", path, 2, 2, lang,
                    meta={"methods": ["GET"], "path": f"/api/things{i}/{{id}}",
                          "handler": f"{cls}::m0", "framework": "laravel"},
                ))
            if lang == "typescript":
                edges.append(Edge("api_call", f"{cls}.m0", f"GET /api/things{i - 1}/{{id}}", path, 9,
                                  None, None, {"method": "GET", "url": f"/api/things{i - 1}/{{id}}"}))
            edges.append(Edge("import", path, f"mod00/pkg0/file_{(i * 7) % n_files:05d}", path, 1,
                              f"mod{(i * 7) % 40:02d}/pkg0/file_{(i * 7) % n_files:05d}.php", None))
            edges.extend(Edge("call", f"{cls}.m{j}", f"helper{j}", path, 6 + j * 10) for j in range(8))
        records.append(FileRecord(path, lang, 4000, 100, f"{i:064x}", f"{lang}-regex",
                                  tuple(symbols), tuple(edges)))
    records.sort(key=lambda rec: rec.path)
    return RepoIndex(root="/synthetic", files=tuple(records))


def test_render_outline_respects_budget_and_degrades_gracefully(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    full = index.render_outline(max_tokens=100_000)
    assert full.startswith(f"# repo outline: 20 files, {index.stats['symbols']} symbols")
    assert f"(digest {index.digest[:12]})" in full
    assert "detail reduced" not in full
    for expected in (
        "src/Models/User.php [php 20L]",
        "class User extends Model implements HasName L8-20  # A user row.",
        "public static function find(\\PDO $pdo, int $id): ?self L14-19",
        "imports: App\\Contracts\\HasName -> src/Contracts/HasName.php",
        "tables: SELECT users -> db/schema.sql",
        "api: GET /api/users/{id} -> routes/api.php::GET /api/users/{id}",
        "fk: orders(user_id) -> users(id)",
        "handlers: onClick:onDelete",
        "hooks: useState, useEffect",
        "calls App\\Http\\UserController::show: App\\Models\\User::find, App\\Support\\slugify",
    ):
        assert expected in full, expected
    assert index.render_outline(max_tokens=estimate_tokens(full) + 100) == full
    previous_len = len(full) + 1
    for budget in (1800, 1500, 800, 400, 200, 100, 50, 20, 5, 1):
        text = index.render_outline(max_tokens=budget)
        assert estimate_tokens(text) <= budget, budget
        assert len(text) <= budget * CHARS_PER_TOKEN
        assert len(text) <= previous_len  # detail only ever shrinks with the budget
        previous_len = len(text)
        if budget >= 100:
            assert text.splitlines()[0] == full.splitlines()[0]
            assert text.splitlines()[1].startswith("# detail reduced to fit the token budget")
    assert index.render_outline(max_tokens=0) == ""
    mid = index.render_outline(max_tokens=1600)
    assert all(rec.path in mid for rec in index.files)  # every file is still listed
    assert "files by level:" in mid.splitlines()[1] and "full" in mid.splitlines()[1]
    assert "TABLE users (id INT PK, name VARCHAR(100))" in mid
    assert "calls App\\Http\\UserController::show" not in mid  # call edges go first
    # the most depended-upon file keeps detail while leaves are reduced to bare paths
    tight = index.render_outline(max_tokens=400).splitlines()
    assert tight[1].endswith("files by level: 14 names, 6 path only")
    assert "db/schema.sql [sql 9L]" in tight and "  users, orders" in tight
    assert "web/src/index.js" in tight and "svc/app/main.py" in tight
    paths_only = index.render_outline(max_tokens=200).splitlines()
    assert paths_only[2:] == [rec.path for rec in index.files]
    rollup = index.render_outline(max_tokens=100)  # not even the path list fits
    assert "file counts per directory" in rollup and "src/ (6 files)" in rollup
    assert "lib/ (1 file)" in rollup
    assert index.render_outline(max_tokens=60).splitlines()[-1] == "... (+4 more lines)"
    assert index.render_outline(1500) == index.render_outline(1500)


def test_render_outline_paths_filter(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    only_svc = index.render_outline(max_tokens=5000, paths=["svc"])
    assert only_svc.startswith("# repo outline: 4 files")
    assert "svc/app/main.py" in only_svc and "routes/api.php [" not in only_svc
    single = index.render_outline(max_tokens=5000, paths="db/schema.sql")
    assert single.startswith("# repo outline: 1 files") and "TABLE orders" in single
    globbed = index.render_outline(max_tokens=5000, paths=["web/src/*.js", "lib\\Mailer.php"])
    assert globbed.startswith("# repo outline: 3 files")  # fnmatch: * also crosses "/"
    assert "lib/Mailer.php" in globbed and "web/src/index.js" in globbed
    assert index.render_outline(paths=["no/such/dir"]).startswith("# repo outline: 0 files")


def test_to_summary_shape_and_content(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    summary = index.to_summary(max_tokens=200_000)
    assert json.loads(json.dumps(summary)) == summary  # JSON-serializable as-is
    assert set(summary) == {
        "schema_version", "digest", "max_tokens", "stats", "files", "routes", "tables",
        "api_calls", "truncated", "omitted",
    }
    assert summary["schema_version"] == SCHEMA_VERSION and summary["digest"] == index.digest
    assert summary["truncated"] is False and not any(summary["omitted"].values())
    assert summary["stats"] == {
        "files": 20, "symbols": index.stats["symbols"], "edges": index.stats["edges"],
        "resolved_edges": index.stats["resolved_edges"],
        "languages": {"javascript": 3, "php": 8, "python": 4, "sql": 2, "typescript": 3},
    }
    files = {entry["path"]: entry for entry in summary["files"]}
    assert list(files) == [rec.path for rec in index.files]  # path order
    user = files["src/Models/User.php"]
    assert (user["language"], user["lines"]) == ("php", 20)
    assert user["deps"] == index.dependencies_of("src/Models/User.php")
    (cls,) = [s for s in user["symbols"] if s["kind"] == "class"]
    assert cls["sig"] == "class User extends Model implements HasName"
    assert cls["line"] == 8 and cls["doc"] == "A user row."
    assert "public function name(): string" in cls["members"]  # member signatures
    assert "public const TABLE = 'users'" in cls["members"]  # minor symbols
    assert any(m.startswith("sql prepare: SELECT id, name FROM users") for m in cls["members"])
    assert "symbols" not in files["svc/app/__init__.py"]  # optional keys are simply absent
    routes = {(tuple(r["methods"]), r["path"]): r for r in summary["routes"]}
    show = routes[(("GET",), "/api/users/{id}")]
    assert show == {
        "methods": ["GET"], "path": "/api/users/{id}",
        "handler": "App\\Http\\UserController::show", "file": "routes/api.php", "line": 6,
        "handler_file": "src/Http/UserController.php",
    }
    assert routes[(("GET",), "/py/health")]["file"] == "svc/app/main.py"
    assert "handler_file" not in routes[(("GET",), "/api/orders")]
    tables = {t["name"]: t for t in summary["tables"]}
    assert tables["orders"] == {
        "name": "orders", "file": "db/schema.sql", "line": 5,
        "columns": ["id INT PK", "user_id INT", "total DECIMAL(10,2)"], "primary_key": ["id"],
        "foreign_keys": [  # gathered across every file that constrains the table
            "(coupon_id) -> coupons(id)", "(region_id) -> regions(id)", "(user_id) -> users(id)",
        ],
    }
    assert set(tables) == {"users", "orders", "coupons"}
    calls = {c["call"]: c for c in summary["api_calls"]}
    assert calls["GET /py/health"] == {
        "call": "GET /py/health", "file": "web/src/api/client.ts", "line": 11,
        "route_file": "svc/app/main.py", "route": "GET /py/health",
    }
    assert "route_file" not in calls["GET /nowhere/at/all"]
    # routes / tables are not repeated in the per-file symbol lists
    assert not [s for s in files["db/schema.sql"].get("symbols", []) if s["kind"] == "table"]
    # restricted to a path prefix
    scoped = index.to_summary(max_tokens=50_000, paths=["svc"])
    assert [entry["path"] for entry in scoped["files"]] == [
        "svc/app/__init__.py", "svc/app/db.py", "svc/app/main.py", "svc/app/models.py",
    ]
    assert scoped["stats"]["languages"] == {"python": 4} and scoped["tables"] == []
    assert json.dumps(index.to_summary(3000)) == json.dumps(index.to_summary(3000))  # deterministic


def test_to_summary_drops_least_important_detail_first(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, POLYREPO))
    generous = index.to_summary(max_tokens=200_000)
    minor_kinds = scout._SUMMARY_MINOR_KINDS
    total = {
        "files": 20, "routes": len(generous["routes"]), "tables": 3,
        "api_calls": len(generous["api_calls"]),
        "deps": sum(len(index.dependencies_of(rec.path)) for rec in index.files),
        "symbols": sum(1 for e in generous["files"] for s in e.get("symbols", ())
                       if s["kind"] not in minor_kinds),
        "detail": sum(len(s.get("members", ())) for e in generous["files"]
                      for s in e.get("symbols", ()))
        + sum(1 for e in generous["files"] for s in e.get("symbols", ())
              if s["kind"] in minor_kinds),
    }
    assert total["routes"] == 21 and total["symbols"] == 32 and total["detail"] == 31
    for budget in (200_000, 3200, 3000, 2500, 2000, 1500, 1000, 600, 400, 250, 150):
        summary = index.to_summary(max_tokens=budget)
        assert _summary_tokens(summary) <= budget, budget
        omitted = summary["omitted"]
        assert summary["truncated"] is any(omitted.values())
        assert summary["max_tokens"] == budget
        # the omitted counters are exact
        entries = summary["files"]
        assert len(entries) + omitted["files"] == total["files"]
        assert len(summary["routes"]) + omitted["routes"] == total["routes"]
        assert len(summary["tables"]) + omitted["tables"] == total["tables"]
        assert len(summary["api_calls"]) + omitted["api_calls"] == total["api_calls"]
        assert sum(len(e.get("deps", ())) for e in entries) + omitted["dependencies"] == (
            total["deps"]
        )
        listed = [s for e in entries for s in e.get("symbols", ())]
        majors = sum(1 for s in listed if s["kind"] not in minor_kinds)
        assert majors + omitted["symbols"] == total["symbols"]
        detail = sum(len(s.get("members", ())) for s in listed) + (len(listed) - majors)
        assert detail + omitted["members"] + omitted["minor_symbols"] == total["detail"]
        assert sum(1 for t in summary["tables"] if len(t) > 3) + omitted["table_details"] == 3
        assert sum(1 for s in listed if "doc" in s) + omitted["docs"] == 1
        assert all(e.get("symbols", True) for e in entries)  # no empty placeholders
    # an almost sufficient budget loses strictly the least important tiers first ...
    near = index.to_summary(max_tokens=3000)["omitted"]
    assert near["docs"] == 1 and near["minor_symbols"] > 0
    assert not any(near[key] for key in near if key not in ("docs", "minor_symbols"))
    less = index.to_summary(max_tokens=2500)["omitted"]
    assert (less["members"], less["minor_symbols"], less["docs"]) == (13, 17, 1)
    assert 0 < less["symbols"] < total["symbols"]
    assert not any(less[key] for key in ("files", "routes", "tables", "table_details",
                                         "dependencies", "api_calls"))
    # ... while a far too small one still keeps a slice of every structural tier
    tight = index.to_summary(max_tokens=1000)
    assert tight["truncated"] is True
    assert tight["files"] and tight["routes"] and tight["tables"] and tight["api_calls"]
    assert any("symbols" in entry for entry in tight["files"])
    assert any("deps" in entry for entry in tight["files"])
    assert any("columns" in table for table in tight["tables"])
    assert all(tight["omitted"][key] for key in ("files", "routes", "symbols", "members"))
    # the most depended-upon file is listed for as long as any file is
    assert index.to_summary(max_tokens=250)["files"][0]["path"] == "db/schema.sql"
    # absurd budgets still honour the cap
    tiny = index.to_summary(max_tokens=20)
    assert tiny["truncated"] is True and set(tiny) == {"truncated", "omitted"}
    assert _summary_tokens(tiny) <= 20
    assert index.to_summary(max_tokens=6) == {"truncated": True}
    assert index.to_summary(max_tokens=1) == {} == index.to_summary(max_tokens=0)


def test_outline_and_summary_honour_budget_for_thousands_of_files() -> None:
    """Both hooks stay under their cap (incl. the 90K flash cap) on a 6,000-file index."""
    index = _synthetic_index(6000)
    assert len(index.files) == 6000 and sum(len(f.symbols) for f in index.files) > 40_000
    assert estimate_tokens(index.to_json()) > 20 * FLASH_PROMPT_TOKEN_CAP  # far too big raw
    started = time.perf_counter()
    for budget in (FLASH_PROMPT_TOKEN_CAP, 30_000, 8000, 2000, 500, 120, 40):
        outline = index.render_outline(max_tokens=budget)
        assert estimate_tokens(outline) <= budget, budget
        assert outline.startswith("# repo outline: 6000 files")
        assert "# detail reduced to fit the token budget" in outline or budget < 60
        summary = index.to_summary(max_tokens=budget)
        assert _summary_tokens(summary) <= budget, budget
        assert estimate_tokens(json.dumps(summary, sort_keys=True, separators=(",", ":"))) <= budget
        if budget >= 500:
            assert summary["truncated"] is True and summary["stats"]["files"] == 6000
            assert len(summary["files"]) + summary["omitted"]["files"] == 6000
            assert len(summary["routes"]) + summary["omitted"]["routes"] == 1500
            assert len(summary["tables"]) + summary["omitted"]["tables"] == 1500
            # the budget is actually used, not just respected
            assert _summary_tokens(summary) > budget * 0.9
            assert estimate_tokens(outline) > budget * 0.8
            # no tier starves another: every section keeps a slice of the budget
            assert summary["files"] and summary["tables"] and summary["routes"]
            assert summary["api_calls"]
            assert any("symbols" in entry for entry in summary["files"])
    assert time.perf_counter() - started < 120
    capped = index.render_outline(max_tokens=FLASH_PROMPT_TOKEN_CAP)
    assert capped.count("\n") > 6000  # every file is present at the flash cap
    rollup = index.render_outline(max_tokens=2000)
    assert "file counts per directory" in rollup and "mod00/pkg0/ (" in rollup
    # the most depended-upon file is the first to receive detail
    hub = "mod02/pkg2/file_00002.sql"
    assert len(index.dependents_of(hub)) == 1499
    summary = index.to_summary(max_tokens=30_000)
    assert hub in {entry["path"] for entry in summary["files"]}
    scoped = index.render_outline(max_tokens=3000, paths=["mod07"])
    assert estimate_tokens(scoped) <= 3000 and scoped.startswith("# repo outline: 150 files")


# ======================================================================================
# Gate 10: Error & Configuration Gate
# ======================================================================================


def test_configuration_errors(tmp_path: Path, monkeypatch: Any) -> None:
    assert issubclass(ScoutConfigurationError, ScoutError) and issubclass(ScoutError, Exception)
    with pytest.raises(ScoutConfigurationError, match="not a directory"):
        PolyglotRepoIndexer(tmp_path / "does-not-exist")
    file_path = tmp_path / "file.py"
    file_path.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ScoutConfigurationError, match="not a directory"):
        PolyglotRepoIndexer(file_path)
    with pytest.raises(ScoutConfigurationError, match="unknown indexer option"):
        PolyglotRepoIndexer(tmp_path, no_such_option=1)
    with pytest.raises(ScoutConfigurationError, match="use_tree_sitter"):
        PolyglotRepoIndexer(tmp_path, use_tree_sitter="yes")
    with pytest.raises(ScoutConfigurationError, match="unknown language"):
        PolyglotRepoIndexer(tmp_path, languages={"cobol"})
    with pytest.raises(ScoutConfigurationError):
        PolyglotRepoIndexer(tmp_path, max_file_bytes=0)
    # use_tree_sitter=True without the package is a clear configuration error ...
    monkeypatch.setattr(scout, "_load_tree_sitter_core", lambda: None)
    with pytest.raises(ScoutConfigurationError, match="tree_sitter.*not importable"):
        PolyglotRepoIndexer(tmp_path, use_tree_sitter=True)
    # ... while "auto" silently degrades to the regex extractors
    auto = PolyglotRepoIndexer(tmp_path, use_tree_sitter="auto")
    assert auto.tree_sitter_languages == ()
    _write(tmp_path, {"a.php": "<?php class A {}\n"})
    assert auto.index_file("a.php").extractor == "php-regex"
    # options can be given as an IndexerOptions object, with keyword overrides on top
    opts = IndexerOptions(use_tree_sitter=False, max_file_bytes=10)
    assert PolyglotRepoIndexer(tmp_path, opts).options.max_file_bytes == 10
    assert PolyglotRepoIndexer(tmp_path, opts, max_file_bytes=99).options.max_file_bytes == 99
    with pytest.raises(ScoutError, match="outside the repository root"):
        PolyglotRepoIndexer(tmp_path, opts).index_file(tmp_path.parent / "elsewhere.py")


def test_indexing_a_bad_file_never_raises(tmp_path: Path, monkeypatch: Any) -> None:
    indexer = _indexer(tmp_path)
    garbage = {
        "bad.py": "def broken(:\n  class\n\x0c)))",
        "bad.php": "<?php class { function ( { ]]] 'unterminated",
        "bad.ts": "export class { => ` ${ ((( /* never closed",
        "bad.sql": "CREATE TABLE ( ( ( 'unterminated",
        "notes.txt": "plain text",
    }
    _write(tmp_path, garbage)
    for rel in garbage:
        rec = indexer.index_file(rel)
        assert isinstance(rec, FileRecord) and rec.path == rel
    assert indexer.index_file("bad.py").errors  # the parse error is recorded
    assert indexer.index_file("notes.txt").extractor == scout.EXTRACTOR_SKIPPED
    missing = indexer.index_file("missing.py")
    assert missing.extractor == scout.EXTRACTOR_SKIPPED and "unreadable" in missing.errors[0]
    # an extractor crash is contained: the record carries the error, the run continues
    def boom(text: str, b: Any) -> Any:
        raise RuntimeError("extractor exploded")

    monkeypatch.setattr(scout, "_extract_sql", boom)
    index = indexer.index()
    crashed = index.file("bad.sql")
    assert crashed is not None and crashed.extractor == "sql-failed"
    assert crashed.errors == ("extractor failed: RuntimeError: extractor exploded",)
    assert crashed.symbols == () and len(index.files) == 4
    assert index.stats["files_with_errors"] >= 2


def test_index_source_matches_index_file(tmp_path: Path) -> None:
    """In-memory indexing is byte-for-byte the on-disk result, with the same safety checks."""
    repo = _write(tmp_path, {PHP_PATH: PHP_CONTROLLER, TSX_PATH: TSX_APP, SQL_PATH: SQL_SCHEMA,
                             "svc/service.py": PY_SERVICE})
    indexer = _indexer(repo, max_file_bytes=100_000)
    for rel, text in ((PHP_PATH, PHP_CONTROLLER), (TSX_PATH, TSX_APP), (SQL_PATH, SQL_SCHEMA),
                      ("svc/service.py", PY_SERVICE)):
        on_disk = indexer.index_file(rel)
        assert indexer.index_source(rel, text) == on_disk
        assert indexer.index_source(repo / rel, text.encode("utf-8")) == on_disk
        assert on_disk.symbols and on_disk.sha256 == indexer.index_source(rel, text).sha256
    ghost = indexer.index_source("not/on/disk.py", "def ghost(): pass\n")  # need not exist
    assert ghost.path == "not/on/disk.py" and _names(ghost, "function") == ["ghost"]
    assert indexer.index_source("notes.md", "# hi").extractor == scout.EXTRACTOR_SKIPPED
    big = indexer.index_source("big.py", "x = 1\n" * 30_000)
    assert big.extractor == scout.EXTRACTOR_SKIPPED and big.errors[0].startswith("skipped [too_large]")
    assert indexer.index_source("b.py", b"\x00\x01").errors[0].startswith("skipped [binary]")
    assert indexer.index_source("bad.py", "def (:").errors  # never raises
    with pytest.raises(ScoutError, match="outside the repository root"):
        indexer.index_source(tmp_path.parent / "elsewhere.py", "x = 1\n")


def test_aindex_runs_off_the_event_loop(tmp_path: Path) -> None:
    repo = _write(tmp_path, POLYREPO)

    async def _run() -> tuple[RepoIndex, RepoIndex, int]:
        indexer = _indexer(repo)
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        beat = asyncio.create_task(heartbeat())
        first = await indexer.aindex()
        second = await indexer.aindex(previous=first)
        beat.cancel()
        return first, second, ticks

    first, second, ticks = asyncio.run(_run())
    assert first.to_json() == second.to_json() == _index(repo).to_json()
    assert ticks > 1  # the loop kept running while the index was built on a worker thread


def test_module_is_stdlib_only_and_lazy_about_tree_sitter(tmp_path: Path) -> None:
    """Only stdlib, pydantic and this package are imported; tree-sitter stays unloaded."""
    source = (REPO_ROOT / "backend" / "scout_and_ast.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            imported.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"pydantic", "backend"}
    assert imported <= allowed, imported - allowed
    assert "TODO" not in source
    assert sorted(scout.__all__) == sorted(set(scout.__all__))
    assert all(hasattr(scout, name) for name in scout.__all__)
    for name in ("PolyglotRepoIndexer", "RepoIndex", "FileRecord", "Symbol", "Edge",
                 "ScoutError", "ScoutConfigurationError", "estimate_tokens"):
        assert name in scout.__all__
    probe = (  # the child works in this test's tmp_path, never in the system temp root
        "import sys, pathlib\n"
        "import backend.scout_and_ast as s\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "(root / 'a.php').write_text('<?php class A {}')\n"
        "(root / 'b.ts').write_text('export class B {}')\n"
        "index = s.PolyglotRepoIndexer(root, use_tree_sitter=False).index()\n"
        "assert [f.extractor for f in index.files] == ['php-regex', 'typescript-regex']\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('tree_sitter'))\n"
        "assert not loaded, loaded\n"
    )
    probe_root = tmp_path / "probe-repo"
    probe_root.mkdir()
    done = subprocess.run([sys.executable, "-c", probe, str(probe_root)], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in probe_root.iterdir()) == ["a.php", "b.ts"]


def test_step_2_tests_never_create_files_outside_tmp_path() -> None:
    """Disk hygiene: no step-2 test, nor a child process it starts, uses the system temp root.

    Every temporary file belongs under pytest's ``tmp_path``, which pytest prunes. The check
    is textual on purpose: it also covers source handed to a subprocess as a string.
    """
    makers = ("mkdtemp", "mkstemp", "gettempdir", "gettempdirb", "TemporaryDirectory",
              "NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile")
    banned = re.compile(r"\b(?:" + "|".join(makers) + r")\s*\(")
    for name in ("test_step_2_scout_and_ast.py", "test_step_2_scout_swarm.py"):
        source = (REPO_ROOT / "tests" / name).read_text(encoding="utf-8")
        found = [(source.count("\n", 0, m.start()) + 1, m.group(0)) for m in banned.finditer(source)]
        assert found == [], f"{name}: temporary files outside tmp_path at {found}"


def test_estimate_tokens_matches_router_heuristic() -> None:
    assert CHARS_PER_TOKEN == 3.5
    assert estimate_tokens("") == 0 and estimate_tokens("a") == 1
    assert estimate_tokens("a" * 7) == 2 and estimate_tokens("a" * 8) == 3


# ======================================================================================
# Gate 11: Pathological Input Gate
# ======================================================================================

_MB = 1_000_000
# Stress inputs are megabytes only on request: SCOUT_STRESS_FULL=1 restores the full sizes
# (about three minutes per interpreter); the default run uses the same shapes at a fraction
# of that under the same per-byte time budget plus a growth-ratio check, so a super-linear
# scan is caught at either size.
_STRESS_FULL = os.environ.get("SCOUT_STRESS_FULL", "").strip().lower() in ("1", "true", "yes", "on")
_STRESS_BYTES = _MB if _STRESS_FULL else 60_000


def _stress(full: int, reduced: int | None = None) -> int:
    """A stress-input size or count: ``full`` under SCOUT_STRESS_FULL=1, else ``reduced``
    (default: scaled like the pathological inputs, never below one)."""
    if _STRESS_FULL:
        return full
    return reduced if reduced is not None else max(1, full * _STRESS_BYTES // _MB)


def _best_time(run: Any, repeat: int = 1) -> float:
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - started)
    return best


# every case builds its hostile text for a requested size in characters
_PATHOLOGICAL: dict[str, Any] = {
    "nested_parens": lambda n: "(" * n,
    "nested_mixed": lambda n: "([{" * (n // 3),
    "balanced_nesting": lambda n: "([{" * (n // 6) + "}])" * (n // 6),
    "stray_closers": lambda n: ")]}" * (n // 3),
    "unterminated_double_quote": lambda n: '"' + "a" * n,
    "unterminated_single_quote": lambda n: "x = '" + "b " * (n // 2),
    "unterminated_comment": lambda n: "/* " + "x " * (n // 2),
    "many_unterminated_comments": lambda n: "/* x " * (n // 5),
    "unterminated_template": lambda n: "`${" * (n // 3),
    "single_line_statements": lambda n: "a = 1; " * (n // 7),
    "single_line_calls": lambda n: "a(b.c(" * (n // 6),
    "single_long_word": lambda n: "a" * n,
    "function_nesting": lambda n: "function a(){" * (n // 13),
    "class_openers": lambda n: "class A { m() {" * (n // 15),
    "enum_openers": lambda n: "enum A {" * (n // 8),
    "interface_openers_then_semicolons": lambda n: "interface A {" * 200 + ";" * n,
    "arrow_chains": lambda n: "const a = x =>\n." * (n // 16),
    "generic_brackets": lambda n: "foo<" * (n // 4),
    "create_table_open": lambda n: "CREATE TABLE t (" + "a INT, " * (n // 7),
    "sql_dollar_tags": lambda n: "$$ " * (n // 3),
    "php_heredocs": lambda n: "<<<EOT\n" * (n // 7),
    "php_attributes": lambda n: "#[A(" * (n // 4),
    "type_alias_continuations": lambda n: "type A =\n| " * (n // 11),
    "import_then_blank_lines": lambda n: "import " * 50 + "\n" * n,
    "function_then_blank_lines": lambda n: "function " * 50 + "\n" * n + "(",
    "close_paren_then_spaces": lambda n: "function a()" + " " * n + "x",
    "keywords_with_wide_gaps": lambda n: ("public function a(): " + " " * 4000) * (n // 4021 + 1),
    # declarations whose `<` / `(` never closes, one per line and all on one line
    "unclosed_generic_class_lines": lambda n: "class A<\n" * (n // 9),
    "unclosed_generic_interface_lines": lambda n: "interface A<\n" * (n // 13),
    "unclosed_generic_annotation_lines": lambda n: "let a: <\n" * (n // 9),
    "unclosed_generic_arrow_lines": lambda n: "const f = <\n" * (n // 12),
    "unclosed_generic_heritage_one_line": lambda n: "class A extends B< " * (n // 19),
    "unclosed_generic_annotations_one_line": lambda n: "let a: X<Y, " * (n // 12),
    "declarations_with_call_values_one_line": lambda n: "let a = f() " * (n // 12),
    "unclosed_const_groups": lambda n: "const C =(\n" * (n // 11),
}
_PATHOLOGICAL_TARGETS: tuple[tuple[str, str], ...] = (
    ("a.php", "<?php\n"), ("a.jsx", ""), ("a.tsx", ""), ("a.ts", ""),
    ("db/migrations/001_a.sql", ""), ("a.py", ""),
)
_PATHOLOGICAL_BUDGET_S = 20.0  # per megabyte; typical: around one second per megabyte
_TIMER_FLOOR_S = 0.05  # below this a measurement is noise, not a cost


@pytest.mark.parametrize("case", sorted(_PATHOLOGICAL))
def test_pathological_input_completes_in_bounded_time(tmp_path: Path, case: str) -> None:
    """Nested brackets, unterminated literals / comments, one giant line: megabytes with
    SCOUT_STRESS_FULL=1, otherwise the same shapes scaled down under the same per-byte
    budget plus a growth check (4x the input may not cost 10x the time)."""
    body = _PATHOLOGICAL[case](_STRESS_BYTES)
    quarter = _PATHOLOGICAL[case](_STRESS_BYTES // 4)
    assert len(body) >= _STRESS_BYTES * 0.9 and len(quarter) < len(body) * 0.5
    budget = _PATHOLOGICAL_BUDGET_S * _STRESS_BYTES / _MB + (0.0 if _STRESS_FULL else 1.0)
    indexer = _indexer(tmp_path, skip_minified=False, max_file_bytes=50 * _MB)
    for rel, prefix in _PATHOLOGICAL_TARGETS:
        # in-memory (index_source): megabyte files of hostile-looking text make on-access
        # virus scanners stall the first open for many seconds, which is not ours to measure
        started = time.perf_counter()
        rec = indexer.index_source(rel, prefix + body)
        elapsed = time.perf_counter() - started
        assert elapsed < budget, f"{case} / {rel}: {elapsed:.1f}s"
        assert rec.size_bytes >= len(body) and not rec.extractor.endswith("-failed"), rec.errors
        assert len(rec.symbols) <= scout.DEFAULT_MAX_SYMBOLS_PER_FILE
        assert len(rec.edges) <= scout.DEFAULT_MAX_EDGES_PER_FILE
        assert all(len(text) <= MAX_CAPTURED_CHARS for text in _strings(rec.to_dict()))
        if _STRESS_FULL:
            continue  # the megabyte budget above is the gate; the ratio guards the small run
        small = _best_time(lambda: indexer.index_source(rel, prefix + quarter))
        if elapsed >= max(small, _TIMER_FLOOR_S) * 10:  # confirm before failing on a hiccup
            small = _best_time(lambda: indexer.index_source(rel, prefix + quarter), 3)
            elapsed = _best_time(lambda: indexer.index_source(rel, prefix + body), 3)
        assert elapsed < max(small, _TIMER_FLOOR_S) * 10, (
            f"{case} / {rel}: {small:.3f}s for a quarter of the input, {elapsed:.3f}s for all"
        )


def test_extraction_cost_grows_linearly(tmp_path: Path) -> None:
    """Quadrupling a hostile input must not blow the run time up super-linearly."""
    indexer = _indexer(tmp_path, skip_minified=False, max_file_bytes=50 * _MB)
    small_size = _stress(200_000, 60_000)
    for rel, prefix, unit in (
        ("n.php", "<?php\n", "$a->b(function () { new C([{"),
        ("n.tsx", "", "enum A { class B { x = () => (["),
        ("n.js", "", "const a = b => c(`${d}`, '"),
        ("n.sql", "", "CREATE TABLE t (a INT REFERENCES u(b), "),
    ):
        small_text = prefix + unit * (small_size // len(unit))
        large_text = prefix + unit * (4 * small_size // len(unit))
        small = _best_time(lambda: indexer.index_source(rel, small_text), 2)
        large = _best_time(lambda: indexer.index_source(rel, large_text))
        if large >= max(small, _TIMER_FLOOR_S) * 10:  # confirm before failing on a hiccup
            large = _best_time(lambda: indexer.index_source(rel, large_text), 3)
        # linear would be 4x; allow generous noise but reject quadratic growth (16x)
        assert large < max(small, _TIMER_FLOOR_S) * 10, f"{rel}: {small:.3f}s -> {large:.3f}s"


_UNCLOSED_ANGLE_UNITS: tuple[str, ...] = (
    "class A<", "interface A<", "let a: <", "const f = <", "class A extends B<",
    "export const C: X<", "abstract class A<",
)
_VALID_TS_UNIT = (
    "export class Repo<T> extends Base<T> implements Store<T> {\n"
    "  private items: Map<string, T> = new Map();\n"
    "  get(id: string): T | undefined { return this.items.get(id); }\n"
    "}\n"
    "export const make = <T,>(x: T): Box<T> => ({ value: x });\n"
    "let count: Array<number> = [];\n"
    "interface Store<T> { get(id: string): T | undefined }\n"
)


def test_unclosed_angle_brackets_cost_about_what_valid_code_costs(tmp_path: Path) -> None:
    """Malformed declarations with an unclosed `<` were linear but 50-350x slower than valid
    code (minutes for a 2 MB file): every candidate re-ran a 64-token generic-list scan from
    each of the next 64 `<`. On 250 KB inputs they must now stay within a small constant
    factor of well-formed TypeScript of the same size (always full size: this is the proof)."""
    size = 250_000
    indexer = _indexer(tmp_path, skip_minified=False)
    valid = _VALID_TS_UNIT * (size // len(_VALID_TS_UNIT))
    assert len(indexer.index_source("ok.ts", valid).symbols) > 1000  # real work is measured
    baseline = max(_best_time(lambda: indexer.index_source("ok.ts", valid)), 0.1)
    runs = [(unit, "\n") for unit in _UNCLOSED_ANGLE_UNITS]  # one declaration per line ...
    runs += [(unit, " ") for unit in _UNCLOSED_ANGLE_UNITS[:5]]  # ... and all on one line
    for i, (unit, separator) in enumerate(runs):
        rel = ("m.ts", "m.tsx", "m.js")[i % 3]
        text = (unit + separator) * (size // (len(unit) + 1))
        elapsed = _best_time(lambda: indexer.index_source(rel, text))
        if elapsed >= baseline * 8:  # confirm before failing on a hiccup
            elapsed = _best_time(lambda: indexer.index_source(rel, text), 3)
        where = f"{unit!r} x {len(text)} chars ({separator!r}-separated) as {rel}"
        assert elapsed < baseline * 8, f"{where}: {elapsed:.2f}s vs {baseline:.2f}s valid"
        assert elapsed < 5.0, where  # it used to take 12 to 110 seconds
    # default options: a real file of them (one per line, so it is not skipped as minified)
    default = _indexer(tmp_path)
    text = "class A0<\n" * (size // 10)
    started = time.perf_counter()
    rec = default.index_source("generated.ts", text)
    assert time.perf_counter() - started < 5.0
    assert rec.extractor == "typescript-regex" and rec.symbols == ()
    # the lesser PHP instance: `const C = (` walked 256 unclosed groups from every candidate
    php_valid = "<?php\n" + "const LIMIT = (1 + 2) * [3][0];\n" * (size // 32)
    php_baseline = max(_best_time(lambda: indexer.index_source("ok.php", php_valid)), 0.1)
    for unit in ("const C =(\n", "const C =[\n", "const C = [ "):
        text = "<?php\n" + unit * (size // len(unit))
        elapsed = _best_time(lambda: indexer.index_source("m.php", text))
        if elapsed >= php_baseline * 8:
            elapsed = _best_time(lambda: indexer.index_source("m.php", text), 3)
        assert elapsed < php_baseline * 8, f"{unit!r}: {elapsed:.2f}s vs {php_baseline:.2f}s"


def test_generic_list_table_matches_a_forward_scan(tmp_path: Path) -> None:
    """The precomputed `<...>` table answers exactly what scanning from each `<` would."""
    import random

    def forward(ex: Any, pos: int) -> int:  # the bounded scan the table replaced
        code = ex.code
        if code[pos: pos + 1] != "<":
            return pos
        limit = min(ex.n, pos + scout._JS_ANGLE_SPAN)
        depth, p = 0, pos
        for _ in range(scout._JS_TYPE_MAX_TOKENS):
            m = scout._JS_ANGLE_RE.search(code, p, limit)
            if m is None:
                return pos
            tok = m.group()
            if tok in ("<", ">"):
                depth += 1 if tok == "<" else -1
                p = m.end()
                if depth == 0:
                    return p
            elif tok == "=>":
                p = m.end()
            elif tok == ";":
                return pos
            else:
                p = max(ex.mk.opens.get(m.start(), m.start()), m.start()) + 1
        return pos

    rng = random.Random(7)
    pieces = ["<", "<", ">", ">", "=>", ";", "(", ")", "[", "]", "{", "}", "a", " ", "\n", ",",
              "=", "T", "x" * 50, " " * 120, "<T>", "A<B<C>>", "(a < b)", "=> >"]
    checked = 0
    for _ in range(400):
        source = "".join(rng.choice(pieces) for _ in range(rng.choice([5, 20, 80, 300])))
        ex = scout._JsExtractor(source, scout._FileBuilder("t.ts", "typescript"),
                                typescript=True, jsx=False)
        for pos in range(len(ex.code) + 1):
            assert ex._skip_angle(pos) == forward(ex, pos), (source, pos)
            checked += 1
    assert checked > 30_000
    # what the callers rely on: closed lists are skipped as a unit, open ones are not lists
    rec = _record(tmp_path, "g.ts", (
        "export class Repo<T extends { id: string }> extends Base<Map<string, T>> { get() {} }\n"
        "class Open<\n"
        "export const pick = <T,>(xs: T[]): T => xs[0];\n"
        "const lt = a < b, gt = c > d;\n"
        "let typed: Record<string, Handler> = {}, other = 2;\n"
    ))
    assert _names(rec, "class") == ["Repo"] and _names(rec, "method") == ["Repo.get"]
    assert _edge(rec, "extends", "Base").source == "Repo"
    assert "arrow" in _sym(rec, "pick", "function").modifiers
    assert {"lt", "gt"} <= set(_names(rec, "constant"))
    assert _names(rec, "variable") == ["typed", "other"]


# ======================================================================================
# Gate 12: Optional tree-sitter Gate
# ======================================================================================


def _tree_sitter_indexer(root: Path, **options: Any) -> PolyglotRepoIndexer:
    pytest.importorskip("tree_sitter")
    try:
        return PolyglotRepoIndexer(root, use_tree_sitter=True, **options)
    except ScoutConfigurationError as exc:  # tree_sitter without the grammar packages
        pytest.skip(f"tree-sitter grammars unavailable: {exc}")


def test_auto_mode_matches_environment(tmp_path: Path) -> None:
    """'auto' uses tree-sitter exactly when it is importable; the result is valid either way."""
    _write(tmp_path, {"a.php": "<?php\nclass A { function m() {} }\n"})
    indexer = PolyglotRepoIndexer(tmp_path)  # default: use_tree_sitter="auto"
    assert indexer.options.use_tree_sitter == "auto"
    rec = indexer.index_file("a.php")
    if "php" in indexer.tree_sitter_languages:
        assert rec.extractor == "php-regex+tree-sitter"
    else:
        assert rec.extractor == "php-regex"
    assert _names(rec, "class") == ["A"] and _names(rec, "method") == ["A::m"]


def test_tree_sitter_backend_confirms_structure(tmp_path: Path) -> None:
    indexer = _tree_sitter_indexer(_write(tmp_path, {
        PHP_PATH: PHP_CONTROLLER, TSX_PATH: TSX_APP, "ns/braced.php": PHP_NAMESPACES,
        "web/plain.js": "export class K { go() { return 1; } }\nfunction top() {}\n",
        "svc/mod.py": "def f(): pass\n", "db/s.sql": "CREATE TABLE t (id INT);\n",
    }))
    assert set(indexer.tree_sitter_languages) == {"javascript", "php", "tsx", "typescript"}
    index = indexer.index()
    assert indexer.last_run.tree_sitter_languages == indexer.tree_sitter_languages
    extractors = {rec.path: rec.extractor for rec in index.files}
    assert extractors == {
        PHP_PATH: "php-regex+tree-sitter",
        "ns/braced.php": "php-regex+tree-sitter",
        TSX_PATH: "typescript-regex+tree-sitter",
        "web/plain.js": "javascript-regex+tree-sitter",
        "svc/mod.py": "python-ast",  # tree-sitter is only used for PHP / JS / TS
        "db/s.sql": "sql-regex",
    }
    assert all(not rec.errors for rec in index.files), [rec.errors for rec in index.files]
    regex = _index(tmp_path)  # use_tree_sitter=False
    structural = {"class", "interface", "trait", "enum", "function", "method", "component", "hook"}
    for rec in index.files:
        plain = regex.file(rec.path)
        assert plain is not None
        with_ts = {(s.kind, s.qualified_name, s.line) for s in rec.symbols if s.kind in structural}
        without = {(s.kind, s.qualified_name, s.line) for s in plain.symbols
                   if s.kind in structural}
        assert with_ts >= without, without - with_ts  # everything regex found is confirmed
        added = [s for s in rec.symbols if s.meta.get("source") == "tree-sitter"]
        assert {(s.kind, s.qualified_name, s.line) for s in added} == with_ts - without
        # the semantic facts (routes, SQL, handlers, API calls, imports) come from regex, unchanged
        for kind in ("route", "sql_query", "namespace", "constant", "property", "enum_case"):
            assert _names(rec, kind) == _names(plain, kind), (rec.path, kind)
        for kind in ("import", "export", "api_call", "handler", "sql_table", "route", "call",
                     "extends", "implements", "uses_trait", "uses_hook"):
            assert _edges(rec, kind) == _edges(plain, kind), (rec.path, kind)


def test_tree_sitter_gives_accurate_end_lines(tmp_path: Path) -> None:
    php = (
        "<?php\nnamespace N;\n"
        "class Outer\n{\n"
        "    public function first(): int\n    {\n        return 1;\n    }\n\n"
        "    public function second(array $a): array\n    {\n"
        "        return array_map(function ($x) {\n            return $x;\n        }, $a);\n"
        "    }\n}\n"
        "function tail(): void\n{\n}\n"
    )
    ts = (
        "export class Box<T> {\n  private v: T;\n  constructor(v: T) {\n    this.v = v;\n  }\n"
        "  get value(): T {\n    return this.v;\n  }\n}\n\n"
        "export const make = <T,>(v: T): Box<T> => {\n  return new Box(v);\n};\n\n"
        "export interface Shape {\n  area(): number;\n}\n"
    )
    indexer = _tree_sitter_indexer(_write(tmp_path, {"Outer.php": php, "box.ts": ts}))
    outer = indexer.index_file("Outer.php")
    assert outer.extractor == "php-regex+tree-sitter"
    assert (_sym(outer, "N\\Outer").line, _sym(outer, "N\\Outer").end_line) == (3, 16)
    assert (_sym(outer, "N\\Outer::first").line, _sym(outer, "N\\Outer::first").end_line) == (5, 8)
    second = _sym(outer, "N\\Outer::second")
    assert (second.line, second.end_line) == (10, 15)
    assert (_sym(outer, "N\\tail").line, _sym(outer, "N\\tail").end_line) == (17, 19)
    box = indexer.index_file("box.ts")
    assert box.extractor == "typescript-regex+tree-sitter"
    assert (_sym(box, "Box").line, _sym(box, "Box").end_line) == (1, 9)
    assert (_sym(box, "Box.constructor").line, _sym(box, "Box.constructor").end_line) == (3, 5)
    assert (_sym(box, "Box.value").line, _sym(box, "Box.value").end_line) == (6, 8)
    assert (_sym(box, "Shape").line, _sym(box, "Shape").end_line) == (15, 17)
    assert _sym(box, "make").end_line == 13
    # the regex engine alone reports the same spans here: the two engines agree
    plain = _indexer(tmp_path)
    for rel in ("Outer.php", "box.ts"):
        merged = indexer.index_file(rel)
        alone = plain.index_file(rel)
        assert [(s.qualified_name, s.line, s.end_line) for s in merged.symbols] == [
            (s.qualified_name, s.line, s.end_line) for s in alone.symbols
        ]


def test_tree_sitter_adds_symbols_the_regex_pipeline_cannot_see(tmp_path: Path) -> None:
    """Deterministic merge: tree-sitter-only symbols are added and flagged as such."""
    source = (
        "export const api = {\n  helper: 1,\n};\n"
        "export default class extends Base {\n  run() {\n    return 1;\n  }\n}\n"
        "label: function hoisted() {\n  return 2;\n}\n"
    )
    indexer = _tree_sitter_indexer(_write(tmp_path, {"odd.js": source}))
    first = indexer.index_file("odd.js")
    second = indexer.index_file("odd.js")
    assert first == second and first.extractor == "javascript-regex+tree-sitter"
    names = {(s.kind, s.name) for s in first.symbols}
    assert ("function", "hoisted") in names and ("method", "run") in names
    hoisted = _sym(first, "hoisted", "function")
    assert hoisted.meta == {"source": "tree-sitter"} and hoisted.signature == "function hoisted()"
    assert (hoisted.line, hoisted.end_line) == (9, 11)
    regex_only = _indexer(tmp_path).index_file("odd.js")  # a labelled function: regex skips it
    assert "hoisted" not in {s.name for s in regex_only.symbols}
    assert [s.qualified_name for s in regex_only.symbols] == [
        s.qualified_name for s in first.symbols if not s.meta.get("source")
    ]
    index = indexer.index()
    assert index.to_json() == indexer.index().to_json()
    assert index.to_json() == _tree_sitter_indexer(tmp_path, workers=4).index().to_json()


def test_tree_sitter_is_guarded_against_pathological_input(tmp_path: Path) -> None:
    indexer = _tree_sitter_indexer(tmp_path, skip_minified=False, max_file_bytes=50 * _MB)
    cases = {
        "deep.php": ("<?php\n" + "f([" * _stress(150_000, 20_000), "bracket nesting depth"),
        "deep.ts": ("(" * _stress(500_000, 60_000), "bracket nesting depth"),
        "comment.js": ("/* x " * _stress(200_000, 25_000), "unterminated block comment"),
        "comment.php": ("<?php\n/* " + "x " * _stress(400_000, 50_000),
                        "unterminated block comment"),
    }
    for rel, (text, reason) in cases.items():
        started = time.perf_counter()
        rec = indexer.index_source(rel, text)
        elapsed = time.perf_counter() - started
        assert elapsed < _PATHOLOGICAL_BUDGET_S, f"{rel}: {elapsed:.1f}s"
        assert not rec.extractor.endswith("+tree-sitter")  # regex result kept
        assert any("tree-sitter skipped" in e and reason in e for e in rec.errors), rec.errors
    # hostile but shallow input still goes through tree-sitter and stays bounded
    for rel, text in (
        ("flat.tsx", "<div>" * _stress(100_000, 20_000)),
        ("flat.php", "<?php\n" + "$a->b(1); " * _stress(80_000, 10_000)),
        ("line.js", "a = 1; " * _stress(120_000, 15_000)),
        ("quote.ts", '"' + "a" * _stress(_MB, 150_000)),
    ):
        started = time.perf_counter()
        rec = indexer.index_source(rel, text)
        elapsed = time.perf_counter() - started
        assert elapsed < 3 * _PATHOLOGICAL_BUDGET_S, f"{rel}: {elapsed:.1f}s"
        assert isinstance(rec, FileRecord)


def test_tree_sitter_parse_budget_aborts_cleanly(tmp_path: Path, monkeypatch: Any) -> None:
    """When the operation budget is spent the parse is cancelled and regex results are kept."""
    source = "".join(f"export function f{i}(a, b) {{ return a + b + {i}; }}\n" for i in range(4000))
    indexer = _tree_sitter_indexer(_write(tmp_path, {"many.js": source}))
    normal = indexer.index_file("many.js")
    assert normal.extractor == "javascript-regex+tree-sitter" and not normal.errors
    monkeypatch.setattr(scout, "_TS_OPS_BASE", 1)
    monkeypatch.setattr(scout, "_TS_BYTES_PER_OP_TICK", 10**9)
    starved = indexer.index_file("many.js")
    assert starved.extractor == "javascript-regex"
    assert len(starved.errors) == 1 and "tree-sitter failed" in starved.errors[0]
    assert "regex result kept" in starved.errors[0]
    assert [s.name for s in starved.symbols] == [s.name for s in normal.symbols]
    # the wall-clock cap is the last resort behind the operation budget
    monkeypatch.setattr(scout, "_TS_OPS_BASE", 10**9)
    monkeypatch.setattr(scout, "_TS_DEADLINE_S", 0.0)
    timed_out = indexer.index_file("many.js")
    assert timed_out.extractor == "javascript-regex"
    assert "tree-sitter parse aborted after 0 s" in timed_out.errors[0]
    assert [s.name for s in timed_out.symbols] == [s.name for s in normal.symbols]


def test_incremental_reuse_tracks_the_extractor_mode(tmp_path: Path) -> None:
    repo = _write(tmp_path, {"a.php": "<?php\nclass A {}\n", "b.py": "def b(): pass\n"})
    with_ts = _tree_sitter_indexer(repo)
    first = with_ts.index()
    again = with_ts.index(previous=first)
    assert with_ts.last_run.files_reused == 2 and again.to_json() == first.to_json()
    regex_only = _indexer(repo)
    downgraded = regex_only.index(previous=first)
    # the PHP record was produced by another engine: it is re-parsed, the Python one is reused
    assert (regex_only.last_run.files_parsed, regex_only.last_run.files_reused) == (1, 1)
    assert downgraded.file("a.php").extractor == "php-regex"
    upgraded = with_ts.index(previous=downgraded)
    assert with_ts.last_run.files_parsed == 1
    assert upgraded.file("a.php").extractor == "php-regex+tree-sitter"
    # a file tree-sitter deliberately skips is still reused (the skip is deterministic)
    _write(repo, {"c.js": "/* never closed\nfunction hidden() {}\n"})
    skipped = with_ts.index(previous=upgraded)
    assert with_ts.last_run.files_parsed == 1
    assert skipped.file("c.js").extractor == "javascript-regex"
    assert skipped.file("c.js").errors == (
        "javascript: the block comment opened on line 1 is never closed; the rest of the "
        "file was treated as its content",
        "tree-sitter skipped: unterminated block comment; regex result kept",
    )
    assert with_ts.index(previous=skipped).to_json() == skipped.to_json()
    assert with_ts.last_run.files_reused == 3


# ======================================================================================
# Gate 13: Review Round 1 Regressions (one focused test per confirmed finding)
# ======================================================================================

# ---- Python --------------------------------------------------------------------------


def test_r1_python_stdlib_and_third_party_imports_never_bind_to_repo_modules(
    tmp_path: Path,
) -> None:
    """`import logging` is the stdlib, `import redis` inside app/services/redis.py is not itself."""
    index = _index(_write(tmp_path, {
        "app/__init__.py": "",
        "app/logging.py": "import logging\n\n\ndef setup():\n    logging.basicConfig()\n",
        "app/json.py": "FORMAT = 'compact'\n",
        "app/utils/__init__.py": "",
        "app/utils/types.py": "UserId = int\n",
        "app/services/__init__.py": "",
        "app/services/redis.py": (
            "import redis\nfrom redis import Redis\n\n\nclass Cache(Redis):\n    pass\n"
        ),
        "app/services/billing.py": (
            "import json\nimport logging\nimport types\nfrom typing import Any\n"
            "from app.logging import setup\n"
        ),
        # the genuine src-layout inference keeps working
        "src/pkg/__init__.py": "",
        "src/pkg/core.py": "def f() -> None: ...\n",
        "src/pkg/cli.py": "import pkg.core\nfrom pkg import core\n",
    }))
    redis_rec, billing = index.file("app/services/redis.py"), index.file("app/services/billing.py")
    assert redis_rec is not None and billing is not None
    for rec, target in ((redis_rec, "redis"), (redis_rec, "redis.Redis"), (billing, "json"),
                        (billing, "logging"), (billing, "types"), (billing, "typing.Any")):
        assert _edge(rec, "import", target).resolved_file is None, target
    assert _edge(redis_rec, "extends", "Redis").resolved_file is None
    log_rec = index.file("app/logging.py")
    assert log_rec is not None and _edge(log_rec, "import", "logging").resolved_file is None
    assert _edge(billing, "import", "app.logging.setup").resolved_file == "app/logging.py"
    assert index.dependents_of("app/json.py") == []
    assert index.dependents_of("app/utils/types.py") == []
    assert index.dependencies_of("app/services/billing.py") == ["app/logging.py"]
    cli = index.file("src/pkg/cli.py")
    assert cli is not None
    assert [e.resolved_file for e in _edges(cli, "import")] == ["src/pkg/core.py"] * 2


def test_r1_python_fallback_signature_is_the_header_only(tmp_path: Path) -> None:
    """A `def` without its colon must not drag body lines (secrets) into the signature."""
    rec = _record(tmp_path, "svc/client.py", (
        "import os\n\n\ndef fetch(url, timeout=3)\n"
        "    token = \"sk-live-SECRET-TOKEN-1234567890\"\n"
        "    password = \"hunter2hunter2\"\n"
        "    return os.popen(url).read()\n\n\ndef ok(a, b):\n    return a + b\n"
    ))
    assert rec.extractor == "python-fallback" and "parse error" in rec.errors[0]
    assert _sym(rec, "fetch").signature == "def fetch(url, timeout=3)"
    assert _sym(rec, "ok").signature == "def ok(a, b)"
    assert not any("SECRET" in s.signature or "hunter2" in s.signature for s in rec.symbols)


def test_r1_python_mock_patch_and_cache_decorators_are_not_routes(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "app/main.py": (
            "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n"
            "@app.get('/health')\ndef health():\n    return {'ok': True}\n"
        ),
        "tests/test_signup.py": (
            "import unittest.mock\nfrom unittest import mock\n\n\n"
            "@mock.patch('app.services.mailer.send_email')\ndef test_signup(send_email):\n"
            "    pass\n\n\nclass TestThing:\n"
            "    @unittest.mock.patch('app.db.connect')\n    def test_run(self, connect):\n"
            "        pass\n\n\n@cache.get('users')\ndef cached():\n    pass\n\n\n"
            "@registry.options('fast')\ndef tuned():\n    pass\n"
        ),
    }))
    tests = index.file("tests/test_signup.py")
    assert tests is not None
    assert _names(tests, "route") == [] and _edges(tests, "route") == []
    assert [r["path"] for r in index.to_summary(4000)["routes"]] == ["/health"]


def test_r1_python_fallback_keeps_multiline_and_continued_imports(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "pkg/app/models.py": "class User: pass\n",
        "pkg/app/legacy.py": (
            "import os, sys\nfrom .models import (\n    User,\n    Order as O,\n)\n"
            "from pkg.core.db import connect, \\\n    get_conn\n"
            "from ..shared import (helper,\n                      other_helper)\n"
            "import json; import re\n\n\ndef main():\n    print \"legacy python 2 statement\"\n"
        ),
    }))
    rec = index.file("pkg/app/legacy.py")
    assert rec is not None and rec.extractor == "python-fallback"
    assert _targets(rec, "import") == {
        "os", "sys", "pkg.app.models.User", "pkg.app.models.Order", "pkg.core.db.connect",
        "pkg.core.db.get_conn", "pkg.shared.helper", "pkg.shared.other_helper", "json", "re",
    }
    assert _edge(rec, "import", "pkg.app.models.Order").meta["alias"] == "O"
    assert _edge(rec, "import", "pkg.app.models.User").resolved_file == "pkg/app/models.py"
    assert index.dependencies_of("pkg/app/legacy.py") == ["pkg/app/models.py"]


def test_r1_python_fallback_ignores_code_inside_string_literals(tmp_path: Path) -> None:
    rec = _record(tmp_path, "legacy/tool.py", (
        '"""Legacy command line tool.\n\nUsage::\n\nimport fake_module\n'
        "from fake.pkg import fake_name\nclass FakeClass:\n    def fake_method(self):\n"
        '"""\nimport os\n\nTEMPLATE = \'\'\'\nfrom generated_pkg import generated_name\n'
        "def generated_function(x):\n    return x\n'''\n\n\n"
        "class Tool(object):\n    def run(self, args):\n        print \"running %s\" % args\n"
    ))
    assert rec.extractor == "python-fallback"
    assert {s.qualified_name for s in rec.symbols} == {"TEMPLATE", "Tool", "Tool.run"}
    assert _targets(rec, "import") == {"os"}


def test_r1_python_fallback_closes_scopes_when_a_statement_dedents(tmp_path: Path) -> None:
    rec = _record(tmp_path, "legacy/tool.py", (
        "class Tool(object):\n    def run(self, args):\n        print \"running\"\n\n\n"
        "def helper():\n    return 1\n\n\ntry:\n    from fast import impl\n"
        "except ImportError:\n    def impl():\n        return helper()\n\n\n"
        "class Other(object):\n    def method(self):\n        pass\n\n\n"
        "if __name__ == \"__main__\":\n    def main():\n        Tool().run([])\n    main()\n"
    ))
    assert rec.extractor == "python-fallback"
    for name in ("impl", "main"):
        assert (_sym(rec, name).kind, _sym(rec, name).parent) == ("function", None)
    assert _names(rec, "method") == ["Tool.run", "Other.method"]


def test_r1_python_fallback_extracts_pep695_generic_functions(tmp_path: Path) -> None:
    rec = _record(tmp_path, "lib/generic.py", (
        "class Box[T]:\n    def get(self) -> T: ...\n\n"
        "    def map[U](self, fn: Callable[[T], U]) -> Box[U]: ...\n\n\n"
        "def first[T](xs: list[T]) -> T:\n    return xs[0]\n\n\n"
        "async def gather_all[T, *Ts](*aws: Awaitable[T]) -> list[T]:\n    return []\n\n\n"
        "def broken(:\n    pass\n"
    ))
    assert rec.extractor == "python-fallback"
    assert _sym(rec, "first").signature == "def first[T](xs: list[T]) -> T"
    assert _sym(rec, "Box.map", "method").signature.startswith("def map[U](self, fn:")
    assert "async" in _sym(rec, "gather_all").modifiers


def test_r1_python_fallback_spans_docs_decorators_and_extends_edges(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "legacy/base.py": "class Base(object):\n    pass\n",
        "legacy/tool.py": (
            "from .base import Base\n\nLIMIT = 3\n"
            "class Tool(Base, mixins.Loggable):\n    \"\"\"Tool doc.\"\"\"\n\n"
            "    @staticmethod\n    def run(args):\n        for a in args:\n"
            "            print \"running %s\" % a\n        return 0\n\n"
            "    def stop(self):\n        pass\n\n\ndef main():\n    tool = Tool()\n"
            "    return tool.run([])\n"
        ),
    }))
    rec = index.file("legacy/tool.py")
    assert rec is not None and rec.extractor == "python-fallback"
    spans = {s.qualified_name: (s.line, s.end_line) for s in rec.symbols}
    assert spans == {"LIMIT": (3, 3), "Tool": (4, 14), "Tool.run": (8, 11),
                     "Tool.stop": (13, 14), "main": (17, 19)}
    tool = _sym(rec, "Tool")
    assert tool.doc == "Tool doc." and tool.meta["bases"] == ["Base", "mixins.Loggable"]
    assert _sym(rec, "Tool.run").modifiers == ("staticmethod",)
    assert _sym(rec, "LIMIT", "constant").signature == "LIMIT = 3"
    base = _edge(rec, "extends", "Base")
    assert (base.source, base.resolved_file, base.resolved_symbol) == (
        "Tool", "legacy/base.py", "Base")
    assert _edge(rec, "extends", "mixins.Loggable").source == "Tool"


def test_r1_python_dunder_all_annotated_augmented_and_extended(tmp_path: Path) -> None:
    annotated = _record(tmp_path, "pkg/annotated.py",
                        "__all__: list[str] = [\"Engine\", \"run\"]\n")
    assert _targets(annotated, "export") == {"Engine", "run"} and annotated.symbols == ()
    augmented = _record(tmp_path, "pkg/augmented.py", (
        "__all__ = [\"Engine\"]\n__all__ += [\"run\", \"plugin\"]\n"
        "__all__.extend([\"extra\"])\n__all__.append(\"last\")\n"
    ))
    assert _targets(augmented, "export") == {"Engine", "run", "plugin", "extra", "last"}


def test_r1_python_subscript_constants_are_not_type_aliases(tmp_path: Path) -> None:
    rec = _record(tmp_path, "app/settings.py", (
        "import os\nimport sys\nfrom typing import Callable\n\n"
        "HOSTS = [\"a.example\", \"b.example\"]\nDATABASE_URL = os.environ[\"DATABASE_URL\"]\n"
        "PRIMARY_HOST = HOSTS[0]\nPY_MAJOR = sys.version_info[0]\nDEBUG = CONFIG[\"debug\"]\n"
        "Handler = Callable[[int], None]\nMatrix = list[list[float]]\n"
    ))
    assert _names(rec, "constant") == ["HOSTS", "DATABASE_URL", "PRIMARY_HOST", "PY_MAJOR",
                                       "DEBUG"]
    assert _names(rec, "type_alias") == ["Handler", "Matrix"]


def test_r1_python_calls_to_builtin_named_repo_functions_are_kept(tmp_path: Path) -> None:
    rec = _record(tmp_path, "tpl/engine.py", (
        "from .io import open\nfrom .filters import format as fmt, filter\n\n\n"
        "def compile(source):\n    return source.strip()\n\n\n"
        "def render(path, items):\n    template = compile(open(path))\n"
        "    kept = filter(items)\n    return fmt(template, kept), len(kept)\n"
    ))
    assert _targets(rec, "call", "render") == {"compile", "open", "filter", "fmt"}


def test_r1_python_module_constants_inside_try_if_and_multi_target(tmp_path: Path) -> None:
    rec = _record(tmp_path, "app/compat.py", (
        "import sys\n\nMAX_RETRIES = 3\n\ntry:\n    import ujson as json\n    HAS_UJSON = True\n"
        "except ImportError:\n    import json\n    HAS_UJSON = False\n\n"
        "if sys.platform == \"win32\":\n    DEFAULT_SHELL = \"cmd.exe\"\nelse:\n"
        "    DEFAULT_SHELL = \"/bin/sh\"\n\nLOW, HIGH = 1, 9\nFIRST = SECOND = 0\n\n\n"
        "def run():\n    LOCAL_NOT_A_MODULE_CONSTANT = 1\n    return LOCAL_NOT_A_MODULE_CONSTANT\n"
    ))
    assert _names(rec, "constant") == ["MAX_RETRIES", "HAS_UJSON", "DEFAULT_SHELL", "LOW",
                                       "HIGH", "FIRST", "SECOND"]
    assert _sym(rec, "HAS_UJSON").line == 7  # the first assignment wins


def test_r1_python_pep695_bounds_and_constraints_stay_in_signatures(tmp_path: Path) -> None:
    rec = _record(tmp_path, "lib/generic.py", (
        "def first[T: Comparable](xs: list[T]) -> T:\n    return xs[0]\n\n\n"
        "class Repo[K: (int, str), V: Model]:\n    def get(self, key: K) -> V: ...\n\n\n"
        "type Pair[T: Hashable] = tuple[T, T]\n\n\n"
        "def plain[T, *Ts, **P](x: T) -> T:\n    return x\n"
    ))
    assert rec.extractor == "python-ast" and not rec.errors
    assert _sym(rec, "first").signature == "def first[T: Comparable](xs: list[T]) -> T"
    assert _sym(rec, "Repo").signature == "class Repo[K: (int, str), V: Model]"
    assert _sym(rec, "Pair").signature == "type Pair[T: Hashable] = tuple[T, T]"
    assert _sym(rec, "plain").signature == "def plain[T, *Ts, **P](x: T) -> T"


def test_r1_python_dotted_import_resolves_dotted_base_class(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "pkg/__init__.py": "", "pkg/core.py": "class Engine:\n    pass\n",
        "dotted.py": "import pkg.core\n\n\nclass K(pkg.core.Engine):\n    pass\n",
    }))
    dotted = index.file("dotted.py")
    assert dotted is not None
    ext = _edge(dotted, "extends", "pkg.core.Engine")
    assert (ext.resolved_file, ext.resolved_symbol) == ("pkg/core.py", "Engine")


def test_r1_python_pyi_stub_beside_module_keeps_imports_resolvable(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "src/pkg/__init__.py": "", "src/pkg/other.py": "def thing():\n    return 1\n",
        "src/pkg/other.pyi": "def thing() -> int: ...\n",
        "tools/run.py": "import pkg.other\nfrom pkg import other\nfrom pkg.other import thing\n",
    }))
    run = index.file("tools/run.py")
    assert run is not None
    assert [e.resolved_file for e in _edges(run, "import")] == ["src/pkg/other.py"] * 3


def test_r1_python_pep263_encoding_declarations_are_honoured(tmp_path: Path) -> None:
    sjis = _record(tmp_path, "legacy/sjis_mod.py", (
        "# -*- coding: shift_jis -*-\ndef label():\n"
        "    \"\"\"\u30e9\u30d9\u30eb\u3092\u8fd4\u3059\u3002\"\"\"\n"
        "    return \"\u8868\"\n\n\ndef after():\n    return label()\n"
    ).encode("shift_jis"))
    assert sjis.extractor == "python-ast" and sjis.errors == ()
    assert _sym(sjis, "label").doc == "\u30e9\u30d9\u30eb\u3092\u8fd4\u3059\u3002"
    assert _targets(sjis, "call", "after") == {"label"}
    cp1251 = _record(tmp_path, "legacy/cp1251_mod.py", (
        "# -*- coding: cp1251 -*-\nclass \u0417\u0430\u043f\u0440\u043e\u0441:\n"
        "    \"\"\"\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c.\"\"\"\n"
    ).encode("cp1251"))
    name = "\u0417\u0430\u043f\u0440\u043e\u0441"
    assert _sym(cp1251, name).doc == "\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c."


def test_r1_python_prose_passed_to_text_methods_is_not_sql(tmp_path: Path) -> None:
    rec = _record(tmp_path, "ui/page.py", (
        "import streamlit as st\nfrom sqlalchemy import text\n\n\ndef page(conn, label):\n"
        "    st.text(\"Select a model from the sidebar\")\n"
        "    st.text(\"Update available: download it from the releases page\")\n"
        "    label.text(\"Delete from cart\")\n"
        "    return conn.execute(text(\"SELECT id FROM users WHERE active = 1\"))\n"
    ))
    assert _targets(rec, "sql_table") == {"users"}
    assert all("FROM users" in s.signature for s in rec.symbols if s.kind == "sql_query")


def test_r1_python_fstring_dense_file_is_parsed_without_quadratic_tokenizing(
    tmp_path: Path,
) -> None:
    """CPython 3.12.0-3.12.2 tokenizes f-strings quadratically; the parse must stay bounded."""
    line = "    c.execute(rf'SELECT * FROM {t} WHERE a = {b}')\n"
    # two fields per line: enough lines for fields x size to pass the budget
    count = int((scout._PY_FSTRING_COST_LIMIT / (2 * len(line))) ** 0.5) + 500
    body = "def f(c, t, b):\n" + line * count
    started = time.perf_counter()
    rec = _indexer(tmp_path).index_source("queries.py", body)
    assert time.perf_counter() - started < 10.0
    assert rec.extractor == "python-ast" and _names(rec, "function") == ["f"]
    assert f"python: {2 * count} f-string replacement fields" in rec.errors[0]
    assert _targets(rec, "call") == {"c.execute"}  # structure and literal text survive
    assert _sym(rec, "f#sql@2").meta["verb"] == "SELECT"
    small = _indexer(tmp_path).index_source("few.py", "def g(x):\n    return f'{x}'\n")
    assert small.errors == ()


# ---- PHP -----------------------------------------------------------------------------


def test_r1_php_function_prefixed_calls_are_not_declarations(tmp_path: Path) -> None:
    """`function_exists(...)` / `functions_list()` are calls, not functions named `_exists`."""
    rec = _record(tmp_path, "helpers.php", (
        "<?php\nif (!function_exists('lt_env')) {\n"
        "    function lt_env(string $key): mixed { return $_ENV[$key]; }\n}\n"
        "$len = function_exists('mb_strlen') ? mb_strlen('x') : strlen('x');\n"
        "$names = functions_list();\nfunction &by_ref() {}\n"
        "final class Caps {\n    public function check(): bool {\n        switch (true) {\n"
        "            case function_exists('sodium_crypto_box'): return true;\n"
        "            default: return \\function_exists('random_bytes');\n        }\n    }\n}\n"
    ))
    assert _names(rec, "function") == ["lt_env", "by_ref"]
    assert "by-ref" in _sym(rec, "by_ref").modifiers
    assert {"function_exists", "functions_list"} <= _targets(rec, "call", "helpers.php")
    assert "function_exists" in _targets(rec, "call", "Caps::check")


def test_r1_php_anonymous_class_members_are_not_free_functions(tmp_path: Path) -> None:
    """Laravel migrations: up / down are methods of the anonymous class, never functions."""
    index = _index(_write(tmp_path, {
        "database/migrations/2024_01_01_000000_create_things_table.php": (
            "<?php\nuse Illuminate\\Database\\Migrations\\Migration;\n\n"
            "return new class extends Migration\n{\n    public const TABLE = 'things';\n"
            "    public function up(): void { Schema::create(self::TABLE, fn () => 1); }\n"
            "    public function down(): void { Schema::dropIfExists(self::TABLE); }\n};\n"
        ),
        "app/Support/Lazy.php": (
            "<?php\nnamespace App\\Support;\nfinal class Lazy\n{\n"
            "    public function counter(array $items): \\Countable\n    {\n"
            "        return new class($items) implements \\Countable {\n"
            "            public function __construct(private array $items) {}\n"
            "            public function count(): int { return \\count($this->items); }\n"
            "        };\n    }\n}\n"
        ),
        "app/Reports/total.php": (
            "<?php\nnamespace App\\Reports;\nfunction total(array $rows): int\n{\n"
            "    return count($rows);\n}\n"
        ),
    }))
    mig = index.file("database/migrations/2024_01_01_000000_create_things_table.php")
    lazy, caller = index.file("app/Support/Lazy.php"), index.file("app/Reports/total.php")
    assert mig is not None and lazy is not None and caller is not None
    anon = "class@anonymous:4"
    assert _names(mig, "function") == [] and _names(lazy, "function") == []
    assert _names(mig, "method") == [f"{anon}::up", f"{anon}::down"]
    assert _names(mig, "constant") == [f"{anon}::TABLE"]
    assert _sym(mig, anon, "class").modifiers == ("anonymous",)
    assert _edge(mig, "extends", "Illuminate\\Database\\Migrations\\Migration").source == anon
    nested = "App\\Support\\Lazy::counter::class@anonymous:7"
    assert _names(lazy, "method") == [
        "App\\Support\\Lazy::counter", f"{nested}::__construct", f"{nested}::count",
    ]
    assert _edge(lazy, "implements", "Countable").source == nested
    # the builtin count() elsewhere is not resolved to the anonymous class' method
    assert _edge(caller, "call", "count").resolved_file is None
    assert index.dependencies_of("app/Reports/total.php") == []


def test_r1_php_promoted_properties_with_bracketed_attributes(tmp_path: Path) -> None:
    rec = _record(tmp_path, "src/Dto/UserDto.php", (
        "<?php\nnamespace App\\Dto;\nfinal class UserDto\n{\n    public function __construct(\n"
        "        #[Groups(['user:read', 'user:write'])]\n        #[Assert\\NotBlank]\n"
        "        public readonly string $email,\n"
        "        #[Assert\\Choice(choices: ['admin', 'user'], message: 'Pick [one]')]\n"
        "        public string $role = 'user',\n"
        "        #[ORM\\Column(options: ['default' => 0])] protected int $age = 0,\n"
        "        #[\\SensitiveParameter] private ?string $password = null,\n"
        "        string $plain = 'x',\n    ) {\n    }\n}\n"
    ))
    assert {s.qualified_name: s.signature for s in rec.symbols if s.kind == "property"} == {
        "App\\Dto\\UserDto::$email": "public readonly string $email",
        "App\\Dto\\UserDto::$role": "public string $role",
        "App\\Dto\\UserDto::$age": "protected int $age",
        "App\\Dto\\UserDto::$password": "private ?string $password",
    }


def test_r1_php_symfony_class_level_route_on_invokable_controller(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "src/Controller/HelloAction.php": (
            "<?php\nnamespace App\\Controller;\n"
            "#[Route('/hello/{name}', name: 'hello', methods: ['GET'])]\n"
            "final class HelloAction\n{\n"
            "    public function __invoke(string $name): Response { return new Response($name); }\n"
            "}\n"
        ),
        "src/Controller/ItemController.php": (  # a class-level prefix is not a route by itself
            "<?php\nnamespace App\\Controller;\n#[Route('/api/items')]\n"
            "final class ItemController\n{\n"
            "    #[Route('', methods: [Request::METHOD_GET])]\n"
            "    public function list(): Response {}\n"
            "    #[Route('/{id}', methods: [Request::METHOD_PUT, Request::METHOD_PATCH])]\n"
            "    public function update(int $id): Response {}\n"
            "    public function __invoke(): Response {}\n}\n"
        ),
        "assets/app.js": (
            "export const hello = (n) => fetch(`/hello/${n}`);\n"
            "export const save = (id, body) => fetch(`/api/items/${id}`, { method: 'PUT', body });\n"
        ),
    }))
    hello = index.file("src/Controller/HelloAction.php")
    items, app = index.file("src/Controller/ItemController.php"), index.file("assets/app.js")
    assert hello is not None and items is not None and app is not None
    route = _sym(hello, "GET /hello/{name}", "route")
    assert route.meta["handler"] == "App\\Controller\\HelloAction::__invoke"
    assert route.meta["name"] == "hello"
    # methods given as Request::METHOD_* constants are read, not degraded to ANY
    assert _names(items, "route") == ["GET /api/items", "PUT|PATCH /api/items/{id}"]
    resolved = {e.target: (e.resolved_file, e.resolved_symbol) for e in _edges(app, "api_call")}
    assert resolved == {
        "GET /hello/{n}": ("src/Controller/HelloAction.php", "GET /hello/{name}"),
        "PUT /api/items/{id}": ("src/Controller/ItemController.php", "PUT|PATCH /api/items/{id}"),
    }


def test_r1_php_laravel_controller_groups_prefix_chains_and_qualified_facade(
    tmp_path: Path,
) -> None:
    index = _index(_write(tmp_path, {
        "routes/web.php": (
            "<?php\nuse App\\Http\\Controllers\\OrderController;\n"
            "Route::controller(OrderController::class)->group(function () {\n"
            "    Route::get('/orders/{id}', 'show');\n    Route::post('/orders', 'store');\n});\n"
            "Route::middleware('auth')->prefix('admin')->controller(OrderController::class)"
            "->group(function () {\n    Route::delete('/orders/{id}', 'destroy');\n});\n"
            "Route::prefix('webhooks')->post('/stripe', StripeWebhookController::class);\n"
            "\\Route::group(['prefix' => 'blog'], function () {\n"
            "    \\Route::get('/posts', [PostController::class, 'index']);\n});\n"
            "\\Illuminate\\Support\\Facades\\Route::put('/posts/{post}', 'PostController@update');\n"
        ),
        "app/Http/Controllers/OrderController.php": (
            "<?php\nnamespace App\\Http\\Controllers;\nclass OrderController\n{\n"
            "    public function show(int $id) {}\n    public function store() {}\n"
            "    public function destroy(int $id) {}\n}\n"
        ),
    }))
    rec = index.file("routes/web.php")
    assert rec is not None
    handlers = {s.name: s.meta["handler"] for s in rec.symbols if s.kind == "route"}
    controller = "App\\Http\\Controllers\\OrderController"
    assert handlers == {
        "GET /orders/{id}": f"{controller}::show",
        "POST /orders": f"{controller}::store",
        "DELETE /admin/orders/{id}": f"{controller}::destroy",
        "POST /webhooks/stripe": "StripeWebhookController::__invoke",
        "GET /blog/posts": "PostController::index",
        "PUT /posts/{post}": "PostController::update",
    }
    edge = next(e for e in _edges(rec, "route") if e.source == "DELETE /admin/orders/{id}")
    assert (edge.resolved_file, edge.resolved_symbol) == (
        "app/Http/Controllers/OrderController.php", f"{controller}::destroy")


def test_r1_php_fluent_get_calls_are_routes_only_on_the_route_facade(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "routes/web.php": "<?php\nRoute::get('/dashboard', [DashboardController::class, 'index']);\n",
        "tests/Feature/DashboardRouteTest.php": (
            "<?php\nclass DashboardRouteTest extends TestCase\n{\n"
            "    public function test_guest_is_redirected_from_dashboard_route(): void\n    {\n"
            "        $this->withoutMiddleware()->get('/dashboard')->assertRedirect('/login');\n"
            "    }\n}\n"
        ),
        "app/Providers/SyncServiceProvider.php": (
            "<?php\nclass SyncServiceProvider\n{\n    public function boot(Router $router): void\n"
            "    {\n        $u = Http::withToken($t)->get('https://api.example.com/users', []);\n"
            "    }\n}\n"
        ),
        "app/Billing/ChargeService.php": (  # request accessors / HTTP clients, Slim-shaped
            "<?php\nfinal class ChargeService\n{\n"
            "    public function list(Request $r, StripeClient $api): array\n    {\n"
            "        $page = $r->get('page', 1);\n        $sort = $r->get('sort', 'created_at');\n"
            "        $api->post('charges', $payload = ['amount' => 100]);\n"
            "        return $api->get('/v1/customers', ['query' => ['page' => $page]]);\n"
            "    }\n}\n"
        ),
        "public/index.php": (
            "<?php\n$app->get('/books/{id}', [BookAction::class, 'view']);\n"
            "$app->post('/books', 'BookAction:create');\n"
        ),
        "resources/js/app.js": "fetch('/dashboard');\n",
    }))
    routes = {rec.path: _names(rec, "route") for rec in index.files if _names(rec, "route")}
    assert routes == {
        "public/index.php": ["GET /books/{id}", "POST /books"],
        "routes/web.php": ["GET /dashboard"],
    }
    app = index.file("resources/js/app.js")
    assert app is not None
    assert _edge(app, "api_call", "GET /dashboard").resolved_file == "routes/web.php"


def test_r1_php_sql_comments_and_literals_are_not_tables(tmp_path: Path) -> None:
    rec = _record(tmp_path, "repo.php", (
        "<?php\n$pdo->query(\"\n    SELECT u.id, -- pulled from legacy\n           u.name\n"
        "      FROM users u /* join profile later */\n     WHERE u.source = 'import from csv'\n"
        "       AND u.a IS DISTINCT FROM u.b\n\");\n"
        "$pdo->exec(\"INSERT INTO audit (msg) VALUES ('deleted from archive')\");\n"
        "$pdo->query('SELECT COUNT(*) INTO @total FROM jobs');\n"
    ))
    assert [s.meta["tables"] for s in rec.symbols if s.kind == "sql_query"] == [
        ["users"], ["audit"], ["jobs"],
    ]
    assert _targets(rec, "sql_table") == {"users", "audit", "jobs"}


def test_r1_php_constants_with_long_initializers_are_kept(tmp_path: Path) -> None:
    rows = "\n".join(f"        'country_{i:03d}' => ['name' => 'Country {i}', 'iso' => 'C{i:03d}'],"
                     for i in range(80))
    rec = _record(tmp_path, "src/Data/Countries.php", (
        "<?php\nnamespace App\\Data;\n\nfinal class Countries\n{\n"
        "    public const VERSION = '1.0';\n    public const array MAP = [\n" + rows + "\n    ];\n"
        "    private const SMALL = [1, 2];\n}\n\nconst TOP_BIG = [\n" + rows + "\n];\n"
        "const TOP_SMALL = 1;\n"
    ))
    assert len(rows) > 4000
    assert _names(rec, "constant") == [
        "App\\Data\\Countries::VERSION", "App\\Data\\Countries::MAP",
        "App\\Data\\Countries::SMALL", "App\\Data\\TOP_BIG", "App\\Data\\TOP_SMALL",
    ]
    big = _sym(rec, "App\\Data\\Countries::MAP")
    assert big.line == 7 and len(big.signature) <= MAX_SIGNATURE_CHARS


def test_r1_php_resource_routes_only_except_nested_and_bulk(tmp_path: Path) -> None:
    rec = _record(tmp_path, "routes/api.php", (
        "<?php\nRoute::resource('photos', PhotoController::class)->only(['index', 'show']);\n"
        "Route::apiResource('products', ProductController::class)->except(['destroy', 'update']);\n"
        "Route::apiResource('posts.comments', CommentController::class)->only(['index', 'show']);\n"
        "Route::resources(['tags' => TagController::class]);\n"
    ))
    actions: dict[str, set[str]] = {}
    for sym in rec.symbols:
        cls, _, action = sym.meta["handler"].rpartition("::")
        actions.setdefault(cls, set()).add(action)
    assert actions == {
        "PhotoController": {"index", "show"},
        "ProductController": {"index", "store", "show"},
        "CommentController": {"index", "show"},
        "TagController": {"index", "create", "store", "show", "edit", "update", "destroy"},
    }
    assert [s.name for s in rec.symbols if "Comment" in s.meta["handler"]] == [
        "GET /posts/{post}/comments", "GET /posts/{post}/comments/{id}",
    ]


def test_r1_php_nested_quotes_inside_interpolation_do_not_derail_the_lexer(
    tmp_path: Path,
) -> None:
    rec = _record(tmp_path, "src/View/Greeter.php", (
        "<?php\nnamespace App\\View;\n\nfinal class Greeter\n{\n"
        "    public function greet(): string\n    {\n"
        "        return \"Hello {$this->t(\"it's\")} there\";\n    }\n\n"
        "    public function label(array $row): string\n    {\n"
        "        return \"<b>{$row[\"name\"]}</b>\";\n    }\n\n"
        "    private function t(string $key): string\n    {\n        return $key;\n    }\n}\n\n"
        "function after_class(): void\n{\n}\n"
    ))
    assert {s.qualified_name: (s.line, s.end_line) for s in rec.symbols} == {
        "App\\View": (2, 2),
        "App\\View\\Greeter": (4, 20),
        "App\\View\\Greeter::greet": (6, 9),
        "App\\View\\Greeter::label": (11, 14),
        "App\\View\\Greeter::t": (16, 19),
        "App\\View\\after_class": (22, 24),
    }


def test_r1_php_dnf_typed_members_do_not_call_their_modifiers(tmp_path: Path) -> None:
    rec = _record(tmp_path, "src/Misc/Holder.php", (
        "<?php\nnamespace App\\Misc;\nfinal class Holder\n{\n"
        "    public (A&B)|null $prop = null;\n    public function __construct(\n"
        "        private (A&B)|null $dep = null,\n"
        "        protected readonly (A&B)|null $other = null,\n    ) {\n    }\n}\n"
    ))
    assert _targets(rec, "call") == set()
    assert _sym(rec, "App\\Misc\\Holder::$dep").signature == "private (A&B)|null $dep"


def test_r1_php_comments_never_leak_into_signatures(tmp_path: Path) -> None:
    rec = _record(tmp_path, "src/Domain/Order.php", (
        "<?php\nnamespace App\\Domain;\nclass Order\n"
        "    extends Aggregate // legacy base, see ADR-12\n    implements\n        Payable,\n"
        "        /* temporary */ \\Countable\n{\n    public function __construct(\n"
        "        /** Primary key. */\n        public readonly int $id,\n"
        "        // who placed it\n        private string $customer,\n"
        "        # legacy flag\n        bool $legacy = false,\n    ) {\n    }\n}\n"
    ))
    assert _sym(rec, "App\\Domain\\Order").signature == (
        "class Order extends Aggregate implements Payable, \\Countable"
    )
    assert _sym(rec, "App\\Domain\\Order::__construct").signature == (
        "public function __construct(public readonly int $id, private string $customer, "
        "bool $legacy = false)"
    )


def test_r1_php_identifiers_may_start_with_non_ascii_letters(tmp_path: Path) -> None:
    rec = _record(tmp_path, "src/Enc/Groesse.php", (
        "<?php\nnamespace App\\Enc;\nclass Gr\u00f6\u00dfe\n{\n"
        "    public function \u00fcberpr\u00fcfen(): bool { return true; }\n}\n"
        "class \u00c4rger extends Gr\u00f6\u00dfe\n{\n    public function \u00e4ndern(): void {}\n}\n"
        "function \u00f6ffnen(): void {}\n"
    ))
    assert _names(rec, "class") == ["App\\Enc\\Gr\u00f6\u00dfe", "App\\Enc\\\u00c4rger"]
    assert _names(rec, "method") == [
        "App\\Enc\\Gr\u00f6\u00dfe::\u00fcberpr\u00fcfen", "App\\Enc\\\u00c4rger::\u00e4ndern",
    ]
    assert _names(rec, "function") == ["App\\Enc\\\u00f6ffnen"]
    assert _edge(rec, "extends", "App\\Enc\\Gr\u00f6\u00dfe").source == "App\\Enc\\\u00c4rger"


def test_r1_php_short_open_tags_are_php_code(tmp_path: Path) -> None:
    rec = _record(tmp_path, "legacy/page.inc", (
        "<html>\n<?\nfunction legacy_rows($pdo) {\n"
        "    return $pdo->query(\"SELECT id, name FROM things WHERE active = 1\");\n}\n"
        "class LegacyHelper { function esc($s) { return htmlspecialchars($s); } }\n?>\n"
        "<?xml version=\"1.0\"?>\n<p><?= $title ?></p>\n"
    ))
    assert _names(rec, "function") == ["legacy_rows"]
    assert _names(rec, "method") == ["LegacyHelper::esc"]
    assert _targets(rec, "sql_table") == {"things"}


def test_r1_php_unqualified_function_calls_follow_namespace_semantics(tmp_path: Path) -> None:
    files = {
        "src/A/helpers.php": "<?php\nnamespace App\\A;\nfunction fmt(string $s): string {}\n",
        "src/B/helpers.php": "<?php\nnamespace App\\B;\nfunction fmt(string $s): string {}\n",
        "src/A/Service.php": (
            "<?php\nnamespace App\\A;\nclass Service { function run($s) { return fmt($s); } }\n"
        ),
    }
    for extra in ({}, {"src/global_helpers.php": "<?php\nfunction fmt($s) {}\n"}):
        root = tmp_path / f"case{len(extra)}"
        service = _index(_write(root, {**files, **extra})).file("src/A/Service.php")
        assert service is not None
        call = _edge(service, "call", "fmt")
        assert (call.resolved_file, call.resolved_symbol) == ("src/A/helpers.php", "App\\A\\fmt")


def test_r1_php_sql_append_chains_keep_bounded_text(tmp_path: Path) -> None:
    body = "<?php\n$sql = 'SELECT * FROM t WHERE 1';\n" + "$sql .= ' AND a = 1';\n" * 20_000
    extractor = scout._PhpExtractor(body, scout._FileBuilder("q.php", "php"))
    extractor.run()
    entries = extractor.sql_assigns["sql"]
    assert len(entries[-1][1]) == 8000  # the cap is reached once ...
    assert len(entries) < 1000  # ... and no further copies are kept after it


def test_r1_optional_route_parameters_match_calls_with_and_without_them(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "routes/web.php": (
            "<?php\nRoute::post('/users/{id}/roles/{role?}', [RoleController::class, 'assign']);\n"
            "Route::get('/reports/{year}/{month?}', [RoleController::class, 'report']);\n"
        ),
        "public/index.php": "<?php\n$app->get('/items[/{id}]', ItemsAction::class);\n",
        "web/app.js": (
            "fetch(`/users/${id}/roles`, { method: 'POST' });\n"
            "fetch(`/users/${id}/roles/${role}`, { method: 'POST' });\n"
            "fetch('/reports/2024');\nfetch('/items');\nfetch(`/items/${id}`);\n"
        ),
    }))
    app = index.file("web/app.js")
    assert app is not None
    assert [e.resolved_file for e in _edges(app, "api_call")] == [
        "routes/web.php", "routes/web.php", "routes/web.php", "public/index.php",
        "public/index.php",
    ]


# ---- JavaScript / TypeScript ---------------------------------------------------------


def test_r1_js_semicolonless_functions_after_generic_bang_or_jsx_lines(tmp_path: Path) -> None:
    ts = _record(tmp_path, "src/nosemi.ts", (
        "export type Handler = (req: Request) => Promise<Response>\n\n"
        "export async function handle(req: Request): Promise<Response> {\n"
        "  return fetch('/api/handle')\n}\n\n"
        "const el = document.getElementById('root')!\nfunction mount() { el.focus() }\n\n"
        "let registry: Record<string, Handler>\n"
        "export function register(name: string, h: Handler) { registry[name] = h }\n\n"
        "export type Props = Omit<BaseProps, 'id'>\n"
        "export default function main(props: Props) { return mount() }\n"
    ))
    assert {s.name: s.line for s in ts.symbols if s.kind == "function"} == {
        "handle": 3, "mount": 8, "register": 11, "main": 14,
    }
    assert _edge(ts, "api_call", "GET /api/handle").source == "handle"
    assert {"handle", "register", "Handler", "Props"} <= _targets(ts, "export")
    assert _sym(ts, "registry").signature == "let registry: Record<string, Handler>"
    jsx = _record(tmp_path, "src/Page.jsx", (
        "const Title = () =>\n  <h1>hello</h1>\nexport function Page() { return <Title /> }\n"
    ))
    assert _names(jsx, "component") == ["Title", "Page"]


def test_r1_ts_semicolonless_members_ending_in_a_generic(tmp_path: Path) -> None:
    rec = _record(tmp_path, "store.ts", (
        "export interface State {\n  items: Array<Item>\n  byId: Record<string, Item>\n"
        "  load(): Promise<void>\n  onChange?: Handler<Item>\n  reset(): void\n}\n"
        "export class Store {\n  private timer?: ReturnType<typeof setTimeout>\n"
        "  list = async (): Promise<User[]> => this.http.get<User[]>('/api/users')\n"
        "  items: Array<User>\n  load() { return 1 }\n  cache = new Map<string, User>()\n"
        "  save() { return 2 }\n}\nexport type A = Array<string>\n"
        "export type B = Record<string, A>\nexport const c = new Set<string>()\n"
    ))
    members = {s.qualified_name: s.signature for s in rec.symbols if s.parent}
    assert members == {
        "State.items": "items: Array<Item>",
        "State.byId": "byId: Record<string, Item>",
        "State.load": "load(): Promise<void>",
        "State.onChange": "onChange?: Handler<Item>",
        "State.reset": "reset(): void",
        "Store.timer": "private timer?: ReturnType<typeof setTimeout>",
        "Store.list": "list = async (): Promise<User[]> =>",
        "Store.items": "items: Array<User>",
        "Store.load": "load()",
        "Store.cache": "cache",
        "Store.save": "save()",
    }
    alias = _sym(rec, "A", "type_alias")
    assert (alias.line, alias.end_line, alias.signature) == (
        16, 16, "export type A = Array<string>")
    assert _edge(rec, "api_call", "GET /api/users").source == "Store.list"
    assert not {"load", "reset"} & _targets(rec, "call")


def test_r1_ts_class_heritage_with_object_type_arguments(tmp_path: Path) -> None:
    rec = _record(tmp_path, "web/legacy.tsx", (
        "export class Legacy extends React.Component<{ label: string }, { open: boolean }> {\n"
        "  toggle = () => { const next = !this.state.open; this.setState({ open: next }); };\n"
        "  componentDidMount(): void { fetch('/api/ping'); }\n"
        "  render() { return <p onClick={this.toggle}>{this.props.label}</p>; }\n}\n"
        "export class Plain implements Handler<{ a: 1 }>, Other { run(): void {} }\n"
    ))
    assert _names(rec, "method") == [
        "Legacy.toggle", "Legacy.componentDidMount", "Legacy.render", "Plain.run",
    ]
    assert _names(rec, "property") == [] and _names(rec, "constant") == []
    legacy = _sym(rec, "Legacy", "class")
    assert (legacy.line, legacy.end_line) == (1, 5) and "component" in legacy.modifiers
    assert legacy.signature == (
        "export class Legacy extends React.Component<{ label: string }, { open: boolean }>"
    )
    assert _targets(rec, "implements", "Plain") == {"Handler", "Other"}


def test_r1_js_regex_literal_after_an_arrow(tmp_path: Path) -> None:
    rec = _record(tmp_path, "lex.js", (
        "export function parse(src) {\n  const opens = l => /\\{\\s*$/.test(l);\n"
        "  return src.split('\\n').filter(opens);\n}\n"
        "export function after() { return fetch('/api/after'); }\n"
        "function ticks(xs) { return xs.map(x => /`/.test(x)); }\n"
        "export function tail() { return 1; }\n"
    ))
    assert {s.qualified_name: (s.line, s.end_line) for s in rec.symbols} == {
        "parse": (1, 4), "parse.opens": (2, 2), "after": (5, 5), "ticks": (6, 6),
        "tail": (7, 7),
    }


def test_r1_jsx_text_is_not_lexed_as_code(tmp_path: Path) -> None:
    """URLs, backticks and `/*` in JSX text open no comment / template; the rest stays code."""
    rec = _record(tmp_path, "src/Flow.jsx", (
        "function A({ show, u }) {\n  if (show) {\n"
        "    return <p>{show && <a href={u}>https://example.com</a>}</p>;\n  }\n"
        "  return null;\n}\nexport function B() {\n  return (\n    <p>\n"
        "      Docs at https://example.com/docs or <a onClick={openDocs}>here</a> {renderMore()}\n"
        "    </p>\n  );\n}\n"
        "export function Tick() {\n  return <p>Press the ` key, it's fine {thing()}</p>;\n}\n"
        "export function Globs() {\n  return <><code>dist/*.js</code> files</>;\n}\n"
        "export function Last() { return fetch('/api/last'); }\n"
    ))
    assert rec.errors == ()
    assert {s.qualified_name: (s.line, s.end_line) for s in rec.symbols} == {
        "A": (1, 6), "B": (7, 13), "Tick": (14, 16), "Globs": (17, 19), "Last": (20, 20),
    }
    assert _edge(rec, "handler", "openDocs").source == "B"
    assert _targets(rec, "call", "B") == {"renderMore"}
    assert _targets(rec, "call", "Tick") == {"thing"}
    assert _edge(rec, "api_call", "GET /api/last").source == "Last"


def test_r1_js_unterminated_template_is_reported_not_silent(tmp_path: Path) -> None:
    rec = _record(tmp_path, "tpl.js", "export function a() {}\nconst t = `never closed\nfunction b() {}\n")
    assert _names(rec, "function") == ["a"]
    assert rec.errors == (
        "javascript: the template literal opened on line 2 is never closed; the rest of the "
        "file was treated as its content",
    )


def test_r1_js_string_constants_are_scoped_and_shadowed_by_parameters(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "routes/api.php": "<?php\nRoute::post('/api/login', 'AuthController@login');\n",
        "src/http.js": (
            "const BASE = '/api';\n"
            "export async function login(creds) {\n  const url = '/api/login';\n"
            "  return fetch(url, { method: 'POST', body: creds });\n}\n"
            "export async function request(url, opts) {\n  return fetch(url, opts);\n}\n"
            "export function download(url) {\n  return axios.get(url);\n}\n"
            "export function me(BASE) {\n  return fetch(`${BASE}/me`);\n}\n"
            "export function ping() {\n  return fetch(`${BASE}/ping`);\n}\n"
        ),
    }))
    rec = index.file("src/http.js")
    assert rec is not None
    calls = {e.source: (e.target, e.resolved_file) for e in _edges(rec, "api_call")}
    assert calls == {
        "login": ("POST /api/login", "routes/api.php"),  # the constant of this very function
        "request": ("ANY {url}", None),
        "download": ("GET {url}", None),
        "me": ("GET {BASE}/me", None),  # the parameter shadows the module constant
        "ping": ("GET /api/ping", None),
    }


def test_r1_js_shorthand_and_spread_method_options_are_dynamic(tmp_path: Path) -> None:
    rec = _record(tmp_path, "src/users.js", (
        "export function a(method, body) { return fetch('/api/users', { method, body }); }\n"
        "export function b(method, data) { return axios({ url: '/api/users', method, data }); }\n"
        "export function c(type, data) { return $.ajax({ url: '/api/users', type, data }); }\n"
        "export function d(opts) { return fetch('/api/users', { ...opts }); }\n"
        "export function e(body) { return fetch('/api/users', { body }); }\n"
    ))
    assert {e.source: e.meta["method"] for e in _edges(rec, "api_call")} == {
        "a": "ANY", "b": "ANY", "c": "ANY", "d": "ANY", "e": "GET",
    }


def test_r1_js_handler_and_nesting_scans_stay_linear(tmp_path: Path) -> None:
    indexer = _indexer(tmp_path, skip_minified=False)
    for rel, body in (
        ("on.js", "a.onabc=\n" * 12_000),
        ("ns.ts", "namespace a{f();" * 20_000 + "}" * 20_000),
        ("cls.js", "class A{m(){useX();" * 20_000 + "}}" * 20_000),
    ):
        started = time.perf_counter()
        rec = indexer.index_source(rel, body)
        elapsed = time.perf_counter() - started
        assert elapsed < 8.0 and not rec.extractor.endswith("-failed"), (rel, elapsed)


def test_r1_js_default_exported_function_values_are_symbols(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "arrow.js": (
            "export default async (req, res) => {\n  const r = await fetch('/api/arrow');\n"
            "  return res.json(r);\n};\n"
        ),
        "Memo.jsx": (
            "export default memo(function Card({ title, onPick }) {\n"
            "  return <div onClick={onPick}>{title}</div>;\n});\n"
        ),
        "Fwd.tsx": (
            "export default forwardRef<HTMLInputElement, Props>((props, ref) => {\n"
            "  return <input ref={ref} onChange={props.onChange} />;\n});\n"
        ),
        "cjs.js": "module.exports = function handler(req, res, next) {\n  audit(req);\n};\n",
    }))
    found = {
        rec.path: [(s.kind, s.qualified_name, s.signature) for s in rec.symbols]
        for rec in index.files
    }
    assert found == {
        "Fwd.tsx": [("function", "default",
                     "export default forwardRef<HTMLInputElement, Props>((props, ref) =>")],
        "Memo.jsx": [
            ("component", "Card", "export default memo(function Card({ title, onPick })"),
        ],
        "arrow.js": [("function", "default", "export default async (req, res) =>")],
        "cjs.js": [("function", "handler", "module.exports = function handler(req, res, next)")],
    }
    arrow, memo = index.file("arrow.js"), index.file("Memo.jsx")
    assert arrow is not None and memo is not None
    assert _edge(arrow, "api_call", "GET /api/arrow").source == "default"
    assert _edge(memo, "handler", "onPick").source == "Card"
    assert _targets(arrow, "export") == {"default"}


def test_r1_js_every_declarator_of_a_declaration_list_is_indexed(tmp_path: Path) -> None:
    rec = _record(tmp_path, "old.js", (
        "var cache = {}, first = function (a) { return a; },\n"
        "    second = function (b) { const local = '/x'; return helper(b); },\n    third = 3;\n"
        "export const alpha = () => 1,\n  beta = async (x) => { await fetch('/api/beta'); };\n"
    ))
    assert {s.name: s.kind for s in rec.symbols} == {
        "cache": "variable", "first": "function", "second": "function", "third": "variable",
        "alpha": "function", "beta": "function",
    }
    assert _targets(rec, "export") == {"alpha", "beta"}
    assert _edge(rec, "api_call", "GET /api/beta").source == "beta"
    assert _targets(rec, "call", "second") == {"helper"}


def test_r1_js_private_method_declarations_are_not_calls(tmp_path: Path) -> None:
    rec = _record(tmp_path, "priv.ts", (
        "class Svc {\n  #priv(): number { return 1; }\n  #other(a: string): void {}\n"
        "  pub(): number { return 2; }\n  run(): number { return this.#priv() + this.pub(); }\n}\n"
    ))
    assert [(e.source, e.target) for e in _edges(rec, "call")] == [
        ("Svc.run", "this.#priv"), ("Svc.run", "this.pub"),
    ]


def test_r1_ts_exports_inside_namespaces_are_marked(tmp_path: Path) -> None:
    rec = _record(tmp_path, "types/shims.d.ts", (
        "declare module \"*.svg\" {\n  const content: string;\n  export default content;\n}\n"
        "export namespace API {\n  export const BASE = '/v1';\n"
        "  export function url(p: string): string { return BASE + p; }\n}\n"
        "export interface Real { a: number }\n"
    ))
    assert {e.target: e.meta.get("namespace") for e in _edges(rec, "export")} == {
        "default": "*.svg", "BASE": "API", "url": "API", "API": None, "Real": None,
    }


def test_r1_js_locals_of_callbacks_and_object_methods_are_not_module_symbols(
    tmp_path: Path,
) -> None:
    rec = _record(tmp_path, "server.js", (
        "const app = express();\nexport const MAX_ITEMS = 50;\n"
        "app.get('/users/:id', async (req, res) => {\n  const user = await load(req.params.id);\n"
        "  let attempts = 0;\n  res.json(user);\n});\nexport default {\n  methods: {\n"
        "    async load() { const response = await axios.get('/api/items'); },\n  },\n};\n"
        "describe('thing', () => {\n  const fixture = makeFixture();\n"
        "  it('works', function () { const result = run(fixture); });\n});\n"
        "if (process.env.DEBUG) {\n  var verbose = true;\n}\n"
    ))
    assert [s.name for s in rec.symbols] == ["app", "MAX_ITEMS", "verbose"]


def test_r1_ts_class_expressions_and_typed_arrow_return_types(tmp_path: Path) -> None:
    rec = _record(tmp_path, "shape.ts", (
        "export const Repo = class extends Base {\n"
        "  find(id: string) { return query(id); }\n};\n"
        "export const mk = (a: number): { x: number; y: string } => ({ x: a, y: '' });\n"
        "export const thunk = (a: number): (() => void) => () => { run(a); };\n"
    ))
    assert [(s.kind, s.qualified_name) for s in rec.symbols] == [
        ("class", "Repo"), ("method", "Repo.find"), ("function", "mk"), ("function", "thunk"),
    ]
    assert _edge(rec, "extends", "Base").source == "Repo"
    assert _sym(rec, "mk").signature == (
        "export const mk = (a: number): { x: number; y: string } =>")
    assert _sym(rec, "thunk").signature == "export const thunk = (a: number): (() => void) =>"
    assert _targets(rec, "export") == {"Repo", "mk", "thunk"}


def test_r1_minified_heuristic_ignores_long_string_literals(tmp_path: Path) -> None:
    """Hand-written modules with a data URI / inline SVG are indexed; real bundles are not."""
    path_d = "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5 " * 12
    index = _index(_write(tmp_path, {
        "src/assets.js": (
            "export const LOGO = 'data:image/png;base64," + "QUJD" * 12_000 + "';\n"
            + "".join(f"export function fn{i}(a, b) {{\n  return a + b + {i};\n}}\n"
                      for i in range(40))
        ),
        "src/Icons.jsx": "import React from 'react';\n\n" + "".join(
            f"export const Icon{i} = (props) => <svg viewBox='0 0 24 24' {{...props}}>"
            f"<path d='{path_d}'/></svg>;\n" for i in range(8)
        ),
        "dist/app.js": "var a=1;function b(c){return c+a}" * 2000,
    }))
    assets, icons, bundle = (index.file(p) for p in
                             ("src/assets.js", "src/Icons.jsx", "dist/app.js"))
    assert assets is not None and icons is not None and bundle is not None
    assert len(_names(assets, "function")) == 40
    assert len(_sym(assets, "LOGO").signature) <= MAX_SIGNATURE_CHARS
    assert _names(icons, "component") == [f"Icon{i}" for i in range(8)]
    assert bundle.extractor == "skipped" and bundle.errors[0].startswith("skipped [minified]")


# ---- SQL -----------------------------------------------------------------------------


def _columns(rec: FileRecord, table: str) -> list[str]:
    return list(_sym(rec, table, "table").meta["columns"])


def _fks(rec: FileRecord) -> list[tuple[str, str]]:
    return [(e.source, e.target) for e in _edges(rec, "foreign_key")]


def test_r1_sql_go_batch_separators_end_statements(tmp_path: Path) -> None:
    ssms = _record(tmp_path, "mssql/schema.sql", (
        "SET ANSI_NULLS ON\nGO\nCREATE TABLE [dbo].[Customers](\n\t[Id] [int] NOT NULL\n"
        ") ON [PRIMARY]\nGO\nCREATE TABLE [dbo].[Orders](\n"
        "\t[CustomerId] [int] NOT NULL REFERENCES [dbo].[Customers] ([Id])\n) ON [PRIMARY]\ngo\n"
        "CREATE NONCLUSTERED INDEX [IX_Orders] ON [dbo].[Orders]\n(\n\t[CustomerId] ASC\n)\nGO 2\n"
        "CREATE VIEW [dbo].[vOrders] AS SELECT * FROM [dbo].[Orders]\nGO\n"
    ))
    assert _names(ssms, "table") == ["dbo.Customers", "dbo.Orders"]
    assert _sym(ssms, "dbo.Orders", "table").line == 7
    assert _names(ssms, "index") == ["IX_Orders"] and _names(ssms, "view") == ["dbo.vOrders"]
    assert _fks(ssms) == [("dbo.Orders", "dbo.Customers")]
    semis = _record(tmp_path, "mssql/semis.sql", (
        "CREATE TABLE a (id int);\nGO\nCREATE TABLE b (id int REFERENCES a(id));\nGO\n"
    ))
    assert _names(semis, "table") == ["a", "b"] and _fks(semis) == [("b", "a")]


def test_r1_sql_delimiter_word_inside_a_statement_is_not_a_directive(tmp_path: Path) -> None:
    column = _record(tmp_path, "column.sql", (
        "CREATE TABLE csv_formats (\n    id INT PRIMARY KEY,\n"
        "    delimiter VARCHAR(5) NOT NULL DEFAULT ',',\n    quote_char CHAR(1)\n);\n"
        "CREATE TABLE imports (id INT, format_id INT REFERENCES csv_formats(id));\n"
    ))
    assert _columns(column, "csv_formats") == [
        "id INT PK", "delimiter VARCHAR(5)", "quote_char CHAR(1)"]
    assert _fks(column) == [("imports", "csv_formats")]
    copy = _record(tmp_path, "copy.sql", (
        "CREATE TABLE users (id int);\nCOPY users (id)\nFROM '/tmp/users.csv'\n"
        "DELIMITER ','\nCSV HEADER;\nCREATE TABLE after_copy (id int REFERENCES users(id));\n"
    ))
    assert _names(copy, "table") == ["users", "after_copy"]
    real = _record(tmp_path, "real.sql", (
        "DELIMITER $$\nCREATE TRIGGER trg BEFORE INSERT ON users FOR EACH ROW\nBEGIN\n"
        "  SET NEW.name = TRIM(NEW.name);\nEND$$\nDELIMITER ;\nCREATE TABLE after_block (id int);\n"
    ))
    assert _names(real, "trigger") == ["trg"] and _names(real, "table") == ["after_block"]


def test_r1_sql_strings_and_comments_are_not_table_references(tmp_path: Path) -> None:
    seed = _record(tmp_path, "db/seed.sql", (
        "CREATE TABLE notes (id int, body text);\nCREATE TABLE users (id int);\n"
        "INSERT INTO notes (id, body) VALUES\n  (1, 'Imported from legacy system'),\n"
        "  (2, 'Transfer from savings into checking, then delete from users');\n"
        "UPDATE notes SET body = 'moved from archive' WHERE id = 1; -- copied from old_notes\n"
        "DELETE FROM notes /* join ghosts */ WHERE body = 'x';\n"
    ))
    dml = {(e.target, e.meta["verb"]) for e in _edges(seed, "sql_table")
           if e.meta["verb"] != "CREATE"}
    assert dml == {("notes", "INSERT"), ("notes", "UPDATE"), ("notes", "DELETE")}
    views = _record(tmp_path, "db/views.sql", (
        "CREATE VIEW v_orders AS\n  SELECT * FROM orders -- used to read from orders_v1\n"
        "  WHERE note <> 'FROM ghost_table' /* join legacy_orders later */;\n"
    ))
    assert _sym(views, "v_orders", "view").meta["tables"] == ["orders"]
    assert _targets(views, "sql_table", "v_orders") == {"orders"}


def test_r1_sql_character_varying_types_are_kept(tmp_path: Path) -> None:
    rec = _record(tmp_path, "db/structure.sql", (
        "CREATE TABLE public.users (\n    id bigint NOT NULL,\n"
        "    email character varying(255) DEFAULT ''::character varying NOT NULL,\n"
        "    code character(3) NOT NULL,\n"
        "    tags character varying[] DEFAULT '{}'::character varying[],\n"
        "    title national character varying(40),\n"
        "    legacy varchar(20) CHARACTER SET latin1 NOT NULL\n);\n"
    ))
    assert _columns(rec, "public.users") == [
        "id bigint", "email character varying(255)", "code character(3)",
        "tags character varying[]", "title national character varying(40)",
        "legacy varchar(20)",
    ]


def test_r1_sql_foreign_keys_never_cross_schemas_or_guess_between_candidates(
    tmp_path: Path,
) -> None:
    index = _index(_write(tmp_path, {
        "supabase/migrations/0001_users.sql": (
            "create table public.users (\n"
            "  id uuid primary key references auth.users (id) on delete cascade\n);\n"
        ),
        "db/schema.sql": (
            "CREATE TABLE billing.accounts (id int primary key);\n"
            "CREATE TABLE crm.accounts (id int primary key);\n"
            "CREATE TABLE x (a int REFERENCES accounts(id));\n"
            "CREATE TABLE y (a int REFERENCES crm.accounts(id), u uuid REFERENCES users(id));\n"
        ),
    }))
    supabase, schema = index.file("supabase/migrations/0001_users.sql"), index.file("db/schema.sql")
    assert supabase is not None and schema is not None
    cross = _edge(supabase, "foreign_key", "auth.users")
    assert (cross.resolved_file, cross.resolved_symbol) == (None, None)
    resolved = {(e.source, e.target): e.resolved_symbol for e in _edges(schema, "foreign_key")}
    assert resolved == {
        ("x", "accounts"): None,  # billing.accounts or crm.accounts: never the first one
        ("y", "crm.accounts"): "crm.accounts",
        ("y", "users"): "public.users",  # an unqualified name may bind to the only candidate
    }


def test_r1_sql_server_with_check_add_and_clustered_primary_keys(tmp_path: Path) -> None:
    rec = _record(tmp_path, "db/migrations/001_mssql.sql", (
        "CREATE TABLE [dbo].[Customers](\n\t[Id] [int] IDENTITY(1,1) NOT NULL,\n"
        " CONSTRAINT [PK_Customers] PRIMARY KEY CLUSTERED \n(\n\t[Id] ASC\n"
        ")WITH (PAD_INDEX = OFF) ON [PRIMARY]\n) ON [PRIMARY];\n"
        "CREATE TABLE [dbo].[Lines]([OrderId] [int], [LineNo] [int],\n"
        " CONSTRAINT [PK_Lines] PRIMARY KEY NONCLUSTERED ([OrderId] ASC, [LineNo] ASC));\n"
        "ALTER TABLE [dbo].[Lines]  WITH CHECK ADD  CONSTRAINT [FK_Lines] FOREIGN KEY([OrderId])\n"
        "REFERENCES [dbo].[Customers] ([Id])\nON DELETE CASCADE;\n"
        "ALTER TABLE [dbo].[Lines] WITH NOCHECK ADD CONSTRAINT [FK2] FOREIGN KEY([LineNo]) "
        "REFERENCES [dbo].[Customers] ([Id]);\n"
        "ALTER TABLE [dbo].[Lines] ADD [NewCol] INT NULL, [Other] NVARCHAR(10) NULL, "
        "[RefId] INT NULL REFERENCES [dbo].[U] ([Id]);\n"
    ))
    assert _sym(rec, "dbo.Customers", "table").meta["primary_key"] == ["Id"]
    assert _sym(rec, "dbo.Customers", "table").meta["column_count"] == 1
    assert _sym(rec, "dbo.Lines", "table").meta["primary_key"] == ["OrderId", "LineNo"]
    fks = {e.meta.get("constraint"): e for e in _edges(rec, "foreign_key")}
    assert sorted(k for k in fks if k) == ["FK2", "FK_Lines"]
    assert fks["FK_Lines"].meta["columns"] == ["OrderId"]
    assert fks["FK_Lines"].meta["on_delete"] == "CASCADE" and fks["FK_Lines"].line == 10
    assert fks[None].target == "dbo.U"  # inline REFERENCES of a later ADD item
    assert _names(rec, "column") == ["dbo.Lines.NewCol", "dbo.Lines.Other", "dbo.Lines.RefId"]
    operations = _sym(rec, "migration:001", "migration").meta["operations"]
    assert "ALTER TABLE dbo.Lines ADD FOREIGN KEY (OrderId) -> dbo.Customers" in operations


def test_r1_sql_backslash_before_a_closing_quote(tmp_path: Path) -> None:
    bs = chr(92)
    escape = _record(tmp_path, "pg/escape.sql", (
        "CREATE TABLE t1 (id int, name text);\n"
        "CREATE VIEW v AS SELECT * FROM t1 WHERE name LIKE '%" + bs + "_%' ESCAPE '" + bs + "';\n"
        "CREATE TABLE t2 (id int, p text DEFAULT 'C:" + bs + "Data" + bs + "');\n"
        "CREATE TABLE t3 (id int REFERENCES t1(id), e text DEFAULT E'it" + bs + "'s; x');\n"
    ))
    assert _names(escape, "table") == ["t1", "t2", "t3"] and escape.errors == ()
    assert _sym(escape, "v", "view").meta["tables"] == ["t1"]
    mysql = _record(tmp_path, "mysql/seed.sql", (
        "INSERT INTO m VALUES (1, 'it" + bs + "'s; CREATE TABLE ghost (id int);');\n"
        "CREATE TABLE real_after (id int);\n"
    ))
    assert _names(mysql, "table") == ["real_after"]
    open_ended = _record(tmp_path, "open.sql", "CREATE TABLE a (id int);\nSELECT 'never closed\n")
    assert _names(open_ended, "table") == ["a"]
    assert open_ended.errors[0].startswith("sql: unterminated string literal at line 2")


def test_r1_sql_mysql_view_prefixes_and_executable_comments(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "dump.sql": (
            "CREATE TABLE `users` (`id` int NOT NULL, PRIMARY KEY (`id`));\n"
            "CREATE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` SQL SECURITY DEFINER "
            "VIEW `v_active`  AS SELECT `users`.`id` AS `id` FROM `users` ;\n"
            "CREATE OR REPLACE ALGORITHM = MERGE VIEW v_merge AS SELECT 1 FROM users;\n"
            "/*!50001 CREATE VIEW `v_dump` AS select `users`.`id` AS `id` from `users` */;\n"
            "/*!50503 CREATE TABLE `cond_t` (`user_id` int, CONSTRAINT `fk_c` FOREIGN KEY "
            "(`user_id`) REFERENCES `users` (`id`)) */;\n"
            "/* an ordinary comment: CREATE TABLE ghost (id int); */\n"
        ),
        "q.php": "<?php $pdo->query('SELECT * FROM v_active');\n",
    }))
    dump, php = index.file("dump.sql"), index.file("q.php")
    assert dump is not None and php is not None
    assert _names(dump, "view") == ["v_active", "v_merge", "v_dump"]
    assert _sym(dump, "v_active", "view").meta["tables"] == ["users"]
    assert _names(dump, "table") == ["users", "cond_t"] and _fks(dump) == [("cond_t", "users")]
    assert _edge(php, "sql_table", "v_active").resolved_file == "dump.sql"


def test_r1_sql_alter_table_add_primary_key_reaches_the_index(tmp_path: Path) -> None:
    pg = _record(tmp_path, "db/structure.sql", (
        "CREATE TABLE public.users (id bigint NOT NULL, email text NOT NULL);\n"
        "ALTER TABLE ONLY public.users\n    ADD CONSTRAINT users_pkey PRIMARY KEY (id);\n"
    ))
    users = _sym(pg, "public.users", "table")
    assert users.meta["primary_key"] == ["id"]
    assert users.signature == "TABLE public.users (id bigint PK, email text)"
    pma = _record(tmp_path, "db/dump.sql", (
        "ALTER TABLE `roles`\n  ADD PRIMARY KEY (`id`),\n  ADD UNIQUE KEY `name` (`name`);\n"
    ))  # the table itself lives in another file: the key becomes a symbol of its own
    key = _sym(pma, "roles_pkey", "index")
    assert key.signature == "PRIMARY KEY roles_pkey ON roles (id)"
    assert key.meta == {"table": "roles", "columns": ["id"], "unique": True, "primary": True}
    assert _names(pma, "index") == ["roles_pkey", "name"]


def test_r1_sql_migration_versions_from_directories_and_dated_names(tmp_path: Path) -> None:
    expected = {
        "prisma/migrations/20240115120000_init/migration.sql": ("20240115120000", "init"),
        "prisma/migrations/20240220093000_add_posts/migration.sql": ("20240220093000", "add_posts"),
        "db/migrations/2024-01-15-add-users.sql": ("2024-01-15", "add-users"),
        "db/migrations/2024_05_20_add_index.sql": ("2024_05_20", "add_index"),
        "db/migrations/20240115-120000-create-teams.sql": ("20240115-120000", "create-teams"),
        "db/migrations/20240115T130000_create_roles.sql": ("20240115T130000", "create_roles"),
        "db/migrations/2024_01_15_120000_create_things.sql": ("2024_01_15_120000", "create_things"),
        "db/migrations/0007_add_flag.sql": ("0007", "add_flag"),
    }
    index = _index(_write(tmp_path, {path: "CREATE TABLE t (id int);\n" for path in expected}))
    found = {s.file: (s.meta["version"], s.name) for s in index.iter_symbols("migration")}
    assert found == expected
    assert len({s.qualified_name for s in index.iter_symbols("migration")}) == len(expected)


def test_r1_sql_up_down_name_suffix_alone_is_not_a_migration(tmp_path: Path) -> None:
    files = {
        "queries/sign_up.sql": None, "scripts/clean_up.sql": None,
        "reports/monthly_roll-up.sql": None, "reports/drill_down.sql": None,
        "db/migrate/003_drop_legacy.up.sql": "up", "db/migrate/003_drop_legacy.down.sql": "down",
        "migrations/sqls/20240115120000-create-users-down.sql": "down",
    }
    index = _index(_write(tmp_path, {path: "SELECT 1 FROM users;\n" for path in files}))
    found = {s.file: s.meta["direction"] for s in index.iter_symbols("migration")}
    assert found == {path: direction for path, direction in files.items() if direction}


def test_r1_sql_create_table_like_is_not_a_column(tmp_path: Path) -> None:
    rec = _record(tmp_path, "schema.sql", (
        "CREATE TABLE m (id int PRIMARY KEY);\n"
        "CREATE TABLE m_2024 (LIKE m INCLUDING DEFAULTS INCLUDING CONSTRAINTS);\n"
        "CREATE TABLE audit_copy (LIKE public.m INCLUDING ALL, extra text);\n"
        "CREATE TABLE my_copy LIKE other_m;\n"
    ))
    assert _sym(rec, "m_2024", "table").meta["column_count"] == 0
    assert _columns(rec, "audit_copy") == ["extra text"]
    likes = [e.target for e in _edges(rec, "sql_table") if e.meta["verb"] == "LIKE"]
    assert likes == ["m", "public.m", "other_m"]


def test_r1_sql_quoted_identifiers_with_punctuation(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {"schema.sql": (
        "CREATE TABLE \"Customers & Leads\" (\"ID\" int PRIMARY KEY);\n"
        "CREATE TABLE \"Order #\" (\n  \"ID\" int PRIMARY KEY,\n  \"Qty/Unit\" int NOT NULL,\n"
        "  \"Unit Price ($)\" numeric(10,2),\n"
        "  \"Customer ID\" int REFERENCES \"Customers & Leads\" (\"ID\")\n);\n"
        "CREATE TABLE `stats%` (`hit/miss` int);\nCREATE TABLE [Order Details] ([Line #] int);\n"
        "CREATE TABLE m (s varchar(40) DEFAULT \"x; CREATE TABLE ghost (id int)\");\n"
    )}))
    rec = index.file("schema.sql")
    assert rec is not None
    assert _names(rec, "table") == ["Customers & Leads", "Order #", "stats%", "Order Details", "m"]
    assert _columns(rec, "Order #") == [
        "ID int PK", "Qty/Unit int", "Unit Price ($) numeric(10,2)", "Customer ID int"]
    assert _columns(rec, "Order Details") == ["Line # int"]
    fk = _edge(rec, "foreign_key", "Customers & Leads")
    assert (fk.source, fk.resolved_symbol) == ("Order #", "Customers & Leads")


def test_r1_sql_dollar_inside_an_identifier_is_not_a_dollar_quote(tmp_path: Path) -> None:
    rec = _record(tmp_path, "schema.sql", (
        "CREATE TABLE sys$audit$log (id int);\n"
        "CREATE TABLE after_one (id int REFERENCES sys$audit$log (id));\n"
        "CREATE FUNCTION f() RETURNS int AS $body$ SELECT 1; CREATE TABLE ghost (id int); "
        "$body$ LANGUAGE sql;\nCREATE TABLE after_two (id int);\n"
    ))
    assert _names(rec, "table") == ["sys$audit$log", "after_one", "after_two"]
    assert _names(rec, "function") == ["f"] and _fks(rec) == [("after_one", "sys$audit$log")]


def test_r1_sql_tsql_guarded_ddl_is_found(tmp_path: Path) -> None:
    rec = _record(tmp_path, "db/migrations/001_init.sql", (
        "IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[C]'))\n"
        "CREATE TABLE [dbo].[C] ([Id] INT NOT NULL PRIMARY KEY);\n"
        "IF OBJECT_ID('dbo.O', 'U') IS NULL\nBEGIN\n"
        "    CREATE TABLE dbo.O (Id INT PRIMARY KEY, CId INT REFERENCES dbo.C (Id));\nEND;\n"
        "BEGIN TRANSACTION\nCREATE TABLE dbo.Audit (Id INT PRIMARY KEY);\nCOMMIT;\n"
    ))
    assert _names(rec, "table") == ["dbo.C", "dbo.O", "dbo.Audit"]
    assert _fks(rec) == [("dbo.O", "dbo.C")]
    assert _sym(rec, "migration:001", "migration").meta["operations"] == [
        "CREATE TABLE dbo.C", "CREATE TABLE dbo.O", "CREATE TABLE dbo.Audit",
    ]


def test_r1_sql_many_distinct_delimiters_stay_linear(tmp_path: Path) -> None:
    body = "".join(f"DELIMITER x{i}\n" for i in range(30_000)) + "DELIMITER ;\nCREATE TABLE t (i int);\n"
    started = time.perf_counter()
    rec = _indexer(tmp_path).index_source("d.sql", body)
    assert time.perf_counter() - started < 5.0
    assert _names(rec, "table") == ["t"]


# ---- Walker, resolution and API ------------------------------------------------------


def test_r1_route_resolution_is_indexed_not_quadratic(tmp_path: Path) -> None:
    count = 2000
    repo = _write(tmp_path, {
        "routes/api.php": "<?php\n" + "".join(
            f"Route::get('/api/v1/res{i}/{{id}}/items', 'Res{i}Controller@show');\n"
            for i in range(count)),
        "web/client.js": "".join(
            f"export const load{i} = (id) => fetch(`/api/v1/res{i}/${{id}}/items`);\n"
            for i in range(count)),
    })
    index = _index(repo)
    records = list(index.files)
    started = time.perf_counter()
    resolved = scout._resolve_records(records, [])  # the cross-file pass on its own
    elapsed = time.perf_counter() - started
    calls = [e for rec in resolved for e in rec.edges if e.kind == "api_call"]
    assert len(calls) == count and all(e.resolved_file == "routes/api.php" for e in calls)
    assert elapsed < 5.0, f"resolving {count} calls against {count} routes took {elapsed:.1f}s"


def test_r1_api_calls_without_a_literal_path_segment_stay_unresolved(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "routes/web.php": (
            "<?php\nRoute::get('/', [HomeController::class, 'index']);\n"
            "Route::post('/', [HomeController::class, 'submit']);\n"
            "Route::get('/health', fn () => 'ok');\n"
        ),
        "web/http.js": (
            "export function get(url) { return fetch(url); }\n"
            "export const post = (endpoint, body) => fetch(endpoint, { method: 'POST', body });\n"
            "export const viaAxios = (u) => axios.get(u);\n"
            "const BASE = process.env.API_BASE;\nexport const ping = () => fetch(BASE);\n"
            "export const generic = (resource) => fetch(`/${resource}`);\n"
            "export const home = () => fetch('/');\nexport const health = () => fetch('/health');\n"
        ),
    }))
    rec = index.file("web/http.js")
    assert rec is not None
    assert {e.source: e.resolved_symbol for e in _edges(rec, "api_call")} == {
        "get": None, "post": None, "viaAxios": None, "ping": None, "generic": None,
        "home": "GET /", "health": "GET /health",
    }


def test_r1_parameter_routes_win_over_literal_siblings(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "routes/api.php": (
            "<?php\nRoute::get('/api/users/{id}', [UserController::class, 'show']);\n"
            "Route::get('/api/users/me', [UserController::class, 'me']);\n"
            "Route::resource('photos', PhotoController::class);\n"
        ),
        "app.py": (
            "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/items/{item_id}')\n"
            "def get_item(item_id: int): ...\n\n@app.get('/items/search')\n"
            "def search_items(): ...\n"
        ),
        "web/client.ts": (
            "export const user = (id: number) => fetch(`/api/users/${id}`);\n"
            "export const me = () => fetch('/api/users/me');\n"
            "export const photo = (id: number) => axios.get(`/photos/${id}`);\n"
            "export const create = () => axios.get('/photos/create');\n"
            "export const item = (id: number) => fetch('/items/' + id);\n"
            "export const search = () => fetch('/items/search');\n"
        ),
    }))
    rec = index.file("web/client.ts")
    assert rec is not None
    assert {e.source: e.resolved_symbol for e in _edges(rec, "api_call")} == {
        "user": "GET /api/users/{id}", "me": "GET /api/users/me", "photo": "GET /photos/{id}",
        "create": "GET /photos/create", "item": "GET /items/{item_id}",
        "search": "GET /items/search",
    }


def test_r1_foreign_origin_urls_are_labelled_and_never_suffix_matched(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "routes/api.php": (
            "<?php\nRoute::get('/api/users', [UserController::class, 'index']);\n"
            "Route::post('/v1/charges', [BillingController::class, 'charge']);\n"
        ),
        "web/integrations.ts": (
            "export const ghUsers = () => fetch('https://api.github.com/users');\n"
            "export const charge = (b: unknown) => axios.post('https://api.stripe.com/v1/charges', b);\n"
            "export const dev = () => fetch('http://localhost:8000/api/users');\n"
            "export const local = () => fetch('/api/users');\n"
        ),
    }))
    rec = index.file("web/integrations.ts")
    assert rec is not None
    got = {e.source: (e.resolved_symbol, e.meta.get("resolution")) for e in _edges(rec, "api_call")}
    assert got == {
        "ghUsers": (None, None),  # the path prefix is not guessed on top of an ignored host
        "charge": ("POST /v1/charges", "origin-ignored"),
        "dev": ("GET /api/users", "local-origin"),  # assumed to be served by this repo
        "local": ("GET /api/users", None),
    }
    summary = {c["call"]: c for c in index.to_summary(5000)["api_calls"]}
    assert summary["POST https://api.stripe.com/v1/charges"]["match"] == "origin-ignored"


def test_r1_embedded_sql_in_a_variable_is_analysed_once(tmp_path: Path, monkeypatch: Any) -> None:
    calls = 0
    real = scout._analyze_sql

    def counting(sql: str) -> Any:
        nonlocal calls
        calls += 1
        return real(sql)

    monkeypatch.setattr(scout, "_analyze_sql", counting)
    big_sql = "SELECT " + "col, " * 1500 + "x FROM t"
    indexer = _indexer(tmp_path)
    php = indexer.index_source("q.php", "<?php\n$sql = '" + big_sql + "';\n" + "$db->query($sql);\n" * 400)
    py = indexer.index_source("q.py", "q = '" + big_sql + "'\ndef f(c):\n" + "    c.execute(q)\n" * 400)
    assert calls == 2  # once per file, not once per call site
    assert _targets(php, "sql_table") == {"t"} and _targets(py, "sql_table") == {"t"}
    assert len(_names(php, "sql_query")) == 400


def test_r1_ignore_names_are_exact_on_every_os(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "src/Vendor/Billing/Invoice.php": "<?php\nnamespace Vendor\\Billing;\nclass Invoice {}\n",
        "app/Http/Order.php": "<?php\nuse Vendor\\Billing\\Invoice;\n$i = new Invoice();\n",
        "lib/vendor/third_party.php": "<?php\nclass ThirdParty {}\n",
        "Node_Modules/x.js": "export const x = 1;\n",
    }))
    assert [rec.path for rec in index.files] == [
        "Node_Modules/x.js", "app/Http/Order.php", "src/Vendor/Billing/Invoice.php",
    ]
    order = index.file("app/Http/Order.php")
    assert order is not None
    assert _edge(order, "import", "Vendor\\Billing\\Invoice").resolved_file == (
        "src/Vendor/Billing/Invoice.php")


def test_r1_index_file_accepts_paths_spelled_through_a_linked_root(tmp_path: Path) -> None:
    real = _write(tmp_path / "real_repo", {"pkg/mod.py": "def f(): pass\n"})
    link = tmp_path / "workdir"
    _link_dir(link, real)
    indexer = _indexer(link)
    assert indexer.index_file(link / "pkg" / "mod.py").path == "pkg/mod.py"
    assert indexer.index_source(link / "pkg" / "new.py", "def g(): pass\n").path == "pkg/new.py"
    assert indexer.index_file(real / "pkg" / "mod.py").path == "pkg/mod.py"
    with pytest.raises(ScoutError, match="outside the repository root"):
        indexer.index_file(tmp_path / "elsewhere.py")


def test_r1_extra_ignore_globstar_and_anchored_patterns(tmp_path: Path) -> None:
    repo = _write(tmp_path, {rel: "export function f() {}\n" for rel in (
        "a.test.js", "src/b.test.js", "src/deep/c.test.js", "dist/x.js", "pkg/dist/y.js",
        "src/keep.js",
    )})

    def paths(*patterns: str) -> list[str]:
        return [rec.path for rec in _index(repo, extra_ignore=patterns).files]

    tests = ["a.test.js", "src/b.test.js", "src/deep/c.test.js"]
    assert paths("**/*.test.js") == ["dist/x.js", "pkg/dist/y.js", "src/keep.js"]
    assert paths("**/dist") == [*tests, "src/keep.js"]
    assert paths("**/dist/**") == [*tests, "src/keep.js"]
    assert paths("/dist") == [*tests[:1], "pkg/dist/y.js", *tests[1:], "src/keep.js"]
    assert paths("./dist/") == paths("/dist")
    assert paths("src/deep/**") == ["a.test.js", "dist/x.js", "pkg/dist/y.js", "src/b.test.js",
                                    "src/keep.js"]


def test_r1_followed_links_never_shadow_real_dirs_or_bypass_ignores(tmp_path: Path) -> None:
    repo = _write(tmp_path / "repo", {
        "zreal/other.js": "import thing from './sub/thing';\nexport const o = thing();\n",
        "zreal/sub/thing.js": "export default function thing() {}\n",
        "node_modules/dep/index.js": "export const dep = 1;\n",
    })
    outside = _write(tmp_path / "outside", {
        "shared/util.js": "export const u = 1;\n", "node_modules/pkg/i.js": "export const p = 1;\n",
    })
    _link_dir(repo / "alias", repo / "zreal" / "sub")
    _link_dir(repo / "deps", repo / "node_modules")
    _link_dir(repo / "ext", outside)
    _link_dir(repo / "ext2", outside / "shared")  # a second route into the same directory
    _link_dir(outside / "shared" / "back", repo)  # and a loop back into the repository
    index = _index(repo, follow_symlinks=True)
    assert [rec.path for rec in index.files] == [
        "ext/shared/util.js", "zreal/other.js", "zreal/sub/thing.js",
    ]
    other = index.file("zreal/other.js")
    assert other is not None
    assert _edge(other, "import", "./sub/thing").resolved_file == "zreal/sub/thing.js"


def test_r1_bom_is_stripped_even_when_the_latin1_fallback_is_used(tmp_path: Path) -> None:
    bom = b"\xef\xbb\xbf"
    sql = _record(tmp_path, "a.sql", bom + (
        b"CREATE TABLE users (id INT PRIMARY KEY, city VARCHAR(20) DEFAULT 'Orl\xe9ans');\n"
        b"CREATE TABLE orders (id INT, user_id INT REFERENCES users(id));\n"
    ))
    assert _names(sql, "table") == ["users", "orders"]
    assert sql.errors == ("encoding: not valid UTF-8; decoded as latin-1",)
    py = _record(tmp_path, "a.py", bom + b"import os\n\n# caf\xe9\ndef f():\n    return os.getcwd()\n")
    assert py.extractor == "python-ast" and _targets(py, "call", "f") == {"os.getcwd"}


def test_r1_long_paths_keep_file_level_sources_and_distinct_names(tmp_path: Path) -> None:
    deep = "/".join(f"module_directory_level_{i:02d}_with_a_descriptive_name" for i in range(4))
    indexer = _indexer(tmp_path)
    php = indexer.index_source(f"{deep}/report_queries.php", (
        "<?php\nuse App\\Db;\n$db->query('SELECT 1 FROM orders');\n"
        "$db->query('SELECT 2 FROM users');\n"
    ))
    assert len(php.path) > scout.MAX_NAME_CHARS
    assert {e.source for e in php.edges} == {php.path}
    names = _names(php, "sql_query")
    assert len(set(names)) == 2 and [n[-6:] for n in names] == ["#sql@3", "#sql@4"]
    assert all(len(n) <= scout.MAX_NAME_CHARS for n in names)
    js = indexer.index_source(f"{deep}/client.js", "import a from './x';\nfetch('/api/orders');\n")
    assert {e.source for e in js.edges} == {js.path}


def test_r1_incremental_reuse_respects_changed_extraction_options(tmp_path: Path) -> None:
    repo = _write(tmp_path, {
        "many.py": "".join(f"def f{i}(): pass\n" for i in range(50)),
        "bundle.js": "var a=1;" * 2000,
    })
    small = _index(repo, max_symbols_per_file=10)
    assert len(small.file("many.py").symbols) == 10
    fresh = _index(repo)
    assert _indexer(repo).index(previous=small).to_json() == fresh.to_json()
    capped = _indexer(repo, max_symbols_per_file=10)
    assert capped.index(previous=fresh).to_json() == small.to_json()
    loose = _index(repo, skip_minified=False)
    assert loose.file("bundle.js").extractor == "javascript-regex"
    strict = _indexer(repo)
    assert strict.index(previous=loose).to_json() == fresh.to_json()
    assert strict.index(previous=fresh).to_json() == fresh.to_json()
    assert strict.last_run.files_reused == 1  # the skipped bundle is re-checked, many.py reused


def test_r1_index_source_tolerates_lone_surrogates(tmp_path: Path) -> None:
    raw = b"# caf\xe9 (cp1252 comment)\nimport os\n\ndef price(item):\n    return os.getcwd()\n"
    indexer = _indexer(tmp_path)
    escaped = indexer.index_source("legacy.py", raw.decode("utf-8", errors="surrogateescape"))
    assert escaped == indexer.index_source("legacy.py", raw)  # the original bytes come back
    assert _names(escaped, "function") == ["price"]
    lone = indexer.index_source("emoji.js", "const half = '\ud83d';\nexport function f() {}\n")
    assert _names(lone, "function") == ["f"]
    assert lone.errors == ("encoding: unencodable characters (lone surrogates) replaced",)


def test_r1_value_objects_are_hashable(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {"a.py": "import os\n\n\ndef f():\n    return os.getcwd()\n"}))
    symbols, edges = set(index.iter_symbols()), set(index.iter_edges())
    assert len(symbols) == 1 and len(edges) == 2
    assert hash(index.files[0]) == hash(FileRecord.from_dict(index.files[0].to_dict()))
    edge = next(iter(edges))
    assert edge == Edge.from_dict(edge.to_dict()) and edge in edges


def test_r1_from_dict_reports_malformed_stats_as_index_format_error(tmp_path: Path) -> None:
    payload = _index(_write(tmp_path, {"a.py": "x = 1\n"})).to_dict()
    for stats in ("ab", [[1, 2, 3]], 7):
        with pytest.raises(IndexFormatError):
            RepoIndex.from_dict({**payload, "stats": stats})


def test_r1_use_tree_sitter_accepts_only_auto_or_real_booleans(tmp_path: Path) -> None:
    for bad in (0, 1, "yes", None, "AUTO"):
        with pytest.raises(ScoutConfigurationError, match="use_tree_sitter"):
            PolyglotRepoIndexer(tmp_path, use_tree_sitter=bad)
    assert PolyglotRepoIndexer(tmp_path, use_tree_sitter=False).tree_sitter_languages == ()


def test_r1_js_directory_specifiers_and_declaration_file_swaps(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "web/k.ts": "export default 1;\n", "web/k/index.ts": "export default 2;\n",
        "web/types.d.ts": "export interface T { a: number }\n",
        "web/main.ts": (
            "import k from './k/';\nimport file from './k';\n"
            "import type { T } from './types.js';\n"
        ),
    }))
    main = index.file("web/main.ts")
    assert main is not None
    assert {e.target: e.resolved_file for e in _edges(main, "import")} == {
        "./k/": "web/k/index.ts", "./k": "web/k.ts", "./types.js": "web/types.d.ts",
    }


def test_r1_all_lists_every_public_name() -> None:
    public = {
        name for name, value in vars(scout).items()
        if not name.startswith("_") and getattr(value, "__module__", scout.__name__) == (
            scout.__name__) and not isinstance(value, type(os))
    }
    public -= {"annotations", "logger"}
    # plain constants imported from the model router are not this module's API
    import backend.models_router as router_module

    public -= {
        name for name in public
        if getattr(router_module, name, None) is getattr(scout, name)
    }
    assert public <= set(scout.__all__), sorted(public - set(scout.__all__))
    assert len(scout.__all__) == len(set(scout.__all__))
    assert all(hasattr(scout, name) for name in scout.__all__)


# ---- Optional tree-sitter backend ----------------------------------------------------


def test_r1_tree_sitter_added_symbols_carry_headers_not_bodies(tmp_path: Path) -> None:
    backend = _tree_sitter_indexer(tmp_path)._tree_sitter
    assert backend is not None
    syms, _imports = backend.structure("typescript", (
        "export const second = async (x) => {\n  const leakedBodyMarker = x * 2;\n"
        "  return leakedBodyMarker;\n};\n"
        "class K {\n  handler = (e: Event): void => {\n    console.log(leakedBodyMarker);\n  };\n"
        "  plain = function () { return leakedBodyMarker; };\n}\n"
    ))
    got = {s.qn: (s.kind, s.header, s.modifiers) for s in syms}
    assert got == {
        "second": ("function", "second = async (x) =>", ("export", "async", "arrow")),
        "K": ("class", "class K", ()),
        "K.handler": ("method", "handler = (e: Event): void =>", ("arrow",)),
        "K.plain": ("method", "plain = function ()", ("function-expression",)),
    }


def test_r1_tree_sitter_merge_pairs_same_named_symbols_in_one_pass(tmp_path: Path) -> None:
    indexer = _tree_sitter_indexer(tmp_path, skip_minified=False)
    for rel, body in (
        ("bundle.js", "function f(a){return a.b(1)}" * 14_000),
        ("packed.php", "<?php class A {" + "public function m($a){return $a->b();}" * 10_000 + "}"),
    ):
        started = time.perf_counter()
        rec = indexer.index_source(rel, body)
        elapsed = time.perf_counter() - started
        assert rec.extractor.endswith("+tree-sitter") and elapsed < 8.0, (rel, elapsed)


def test_r1_tree_sitter_skips_recovery_heavy_files_deterministically(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """A large file that is mostly syntax errors is refused by content, not by the clock."""
    indexer = _tree_sitter_indexer(tmp_path)
    body = "<?php\n" + "A::b()" * 30_000
    started = time.perf_counter()
    normal = indexer.index_source("calls.php", body)
    assert time.perf_counter() - started < 5.0
    assert normal.extractor == "php-regex"
    assert normal.errors == (
        "tree-sitter skipped: 1 of 1 sampled regions do not parse (dense syntax errors); "
        "regex result kept",
    )

    class _FastClock:  # the wall clock of a 50x slower machine changes nothing
        def __getattr__(self, name: str) -> Any:
            return getattr(time, name)

        def monotonic(self) -> float:
            return time.monotonic() * 50.0

    monkeypatch.setattr(scout, "time", _FastClock())
    assert indexer.index_source("calls.php", body).to_dict() == normal.to_dict()
    monkeypatch.undo()
    # healthy code of the same size is parsed, and the decision is reused incrementally
    healthy = "<?php\n" + "".join(
        f"function f{i}(array $a): int {{\n    return count($a) + {i};\n}}\n" for i in range(900)
    )
    assert len(healthy) > scout._TS_PROBE_MIN_CHARS
    assert indexer.index_source("ok.php", healthy).extractor == "php-regex+tree-sitter"
    _write(tmp_path, {"calls.php": body})
    first = indexer.index()
    assert indexer.index(previous=first).to_json() == first.to_json()
    assert indexer.last_run.files_reused == 1


# ======================================================================================
# Gate 14: Review Round 2 Regressions (one focused test per confirmed finding)
# ======================================================================================

# ---- Pathological input ----------------------------------------------------------------


def test_r2_php_unclosed_string_interpolations_stay_linear_and_are_reported(
    tmp_path: Path,
) -> None:
    """Every unclosed `{$` used to rescan a 2000-character window (40-150x slower)."""
    indexer = _indexer(tmp_path)
    for name, body in (
        ("brackets.php", '<?php $a = "' + "{$a[" * 100_000),
        ("dollars.php", '<?php $a = "' + "${" * 200_000),
    ):
        started = time.perf_counter()
        rec = indexer.index_source(name, body)
        elapsed = time.perf_counter() - started
        assert elapsed < 6.0, (name, elapsed)
        assert rec.errors == (
            "php: the string literal opened on line 1 is never closed; the rest of the file "
            "was treated as its content",
        )
    # the bound is an allowance of failed-scan characters, not a clock: with none left
    # `{$` is plain text, with some left a nested quote does not close the string
    literal = '"{$row["name"]}" . $x'
    assert scout._scan_php_dq(literal, 0) == len('"{$row["name"]}"')
    assert scout._scan_php_dq(literal, 0, [0]) == len('"{$row["')
    spent = [10_000]
    assert scout._scan_php_dq('"' + "{$a[" * 5000, 0, spent) == -1 and spent[0] <= 0
    # closed interpolations never spend it, however many there are
    healthy = "<?php\n" + "".join(
        f'function f{i}($r) {{ return "x {{$r["k{i}"]}} y" . g{i}($r); }}\n' for i in range(3000)
    ) + "$tail = \"it's {$row[\"name\"]}\"; h_last();\n"
    rec = indexer.index_source("ok.php", healthy)
    assert rec.errors == () and len(_names(rec, "function")) == 3000
    assert "h_last" in _targets(rec, "call")
    # other literal kinds that run to the end of the file are reported as well
    for body in ("<?php\n$a = 'never closed;\nf();\n", "<?php\n$a = <<<EOT\nnever closed\n"):
        assert "is never closed" in indexer.index_source("open.php", body).errors[0]


def test_r2_python_fstring_fields_are_budgeted_by_fields_times_size(tmp_path: Path) -> None:
    """One f-string with thousands of fields on one line stalled CPython 3.12.1 for minutes."""
    indexer = _indexer(tmp_path)
    fields = "{a!r:>{w}}" * 40_000  # 440 KB on a single line
    for name, body, extractor in (
        ("closed.py", f"def g():\n    pass\nx = f'{fields}'\n", "python-ast"),
        ("open.py", f"def g():\n    pass\nx = f'{fields}", "python-fallback"),
    ):
        started = time.perf_counter()
        rec = indexer.index_source(name, body)
        elapsed = time.perf_counter() - started
        assert elapsed < 6.0, (name, elapsed)
        assert rec.extractor == extractor and _names(rec, "function") == ["g"]
        assert rec.errors[0].startswith("python: 80000 f-string replacement fields in ")
    assert _sym(indexer.index_source("closed.py", f"x = f'{fields}'\n"), "x").signature == "x = ..."
    # the cost is fields x size, so the same density in a small file is parsed as written
    # and the calls inside its interpolations are still analysed
    small = indexer.index_source(
        "small.py", "def h(a, w):\n    return f'" + "{fmt(a)!r:>{w}}" * 1500 + "'\n"
    )
    assert small.errors == () and small.extractor == "python-ast"
    assert _targets(small, "call", "h") == {"fmt"}
    # braces that are not replacement fields do not count
    data = "ROWS = [" + "{'a': 1}, " * 150_000 + "]\nNOTE = f'{len(ROWS)} rows'\n"
    assert indexer.index_source("data.py", data).errors == ()
    assert scout._py_fstring_fields("x {a!r:>{w}} {{literal}} {b}", 0, 28) == (3, True)
    # neither a literal glued to another one nor a quote nested in a field (PEP 701, which
    # ends the string early for a pre-3.12 lexer) hides the fields that follow
    assert scout._py_fstring_fields("{a[", 0, 3) == (1, False)
    for name, head in (("glued.py", "x = ''f'"), ("nested.py", "x = f'{a['k']}")):
        started = time.perf_counter()
        rec = indexer.index_source(name, "def g():\n    pass\n" + head + "{a}" * 150_000 + "'\n")
        assert time.perf_counter() - started < 6.0, name
        assert "f-string replacement fields" in rec.errors[0], name
        assert _names(rec, "function") == ["g"]


# ---- Python ----------------------------------------------------------------------------


def test_r2_python_module_level_bindings_are_symbols(tmp_path: Path) -> None:
    """Bare-name aliases, dunder metadata and lower-case module objects used to be dropped."""
    index = _index(_write(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/sub.py": (
            "import logging\nimport typing\nfrom flask import Flask\n\n"
            "UserId = int\nJson = typing.Any\nHandler = BaseHandler\n"
            "__version__ = \"1.2.3\"\n_MAX_ROWS = 50\n"
            "app = Flask(__name__)\nrouter = APIRouter(prefix='/x')\n"
            "log = logging.getLogger(__name__)\nthing = 3\nfirst, second = 1, 2\n"
            "Fallback = handler_fn\n_ = ignored()\n"
            "if True:\n    late = compute()\nthing = 4\nthing += 1\nlog.level = 2\n"
        ),
        "use.py": "from pkg.sub import thing, app, UserId, __version__, missing\n",
        "broken.py": (
            "lower = 3\nUPPER = 4\n__version__ = '1'\nif x:\n    pass\nelse: y = 1\n"
            "def f(:\n    inner = 3\n"
        ),
    }))
    sub = index.file("pkg/sub.py")
    assert sub is not None
    assert [(s.kind, s.signature) for s in sub.symbols] == [
        ("type_alias", "UserId = int"),
        ("type_alias", "Json = typing.Any"),
        ("type_alias", "Handler = BaseHandler"),
        ("constant", "__version__ = '1.2.3'"),
        ("constant", "_MAX_ROWS = 50"),
        ("variable", "app = Flask(__name__)"),
        ("variable", "router = APIRouter(prefix='/x')"),
        ("variable", "log = logging.getLogger(__name__)"),
        ("variable", "thing = 3"),
        ("variable", "first = 1"),
        ("variable", "second = 2"),
        ("variable", "Fallback = handler_fn"),  # a CapWords name bound to a function
        ("variable", "late = compute()"),
    ]
    use = index.file("use.py")
    assert use is not None
    assert {e.target: (e.resolved_file, e.resolved_symbol) for e in _edges(use, "import")} == {
        "pkg.sub.thing": ("pkg/sub.py", "thing"),
        "pkg.sub.app": ("pkg/sub.py", "app"),
        "pkg.sub.UserId": ("pkg/sub.py", "UserId"),
        "pkg.sub.__version__": ("pkg/sub.py", "__version__"),
        "pkg.sub.missing": ("pkg/sub.py", None),
    }
    broken = index.file("broken.py")
    assert broken is not None and broken.extractor == scout.EXTRACTOR_PY_FALLBACK
    assert [(s.kind, s.signature) for s in broken.symbols] == [
        ("variable", "lower = 3"), ("constant", "UPPER = 4"), ("constant", "__version__ = '1'"),
        ("function", "def f(:"),
    ]  # `else: y = 1` is not an assignment to `else`, and locals stay out
    # variables are minor detail: the first thing a tight outline / summary drops
    assert "variable" in scout._OUTLINE_MINOR_KINDS and "variable" in scout._SUMMARY_MINOR_KINDS


# ---- Downstream hooks ------------------------------------------------------------------


def test_r2_to_summary_reports_counts_below_the_skeleton_size() -> None:
    """A budget under the empty skeleton used to yield a bare `{"truncated": true}`."""
    index = _synthetic_index(60)
    full_at = next(n for n in range(1, 400) if "digest" in index.to_summary(max_tokens=n))
    assert full_at > 100
    shapes: list[tuple[str, ...]] = []
    for budget in range(1, full_at):
        summary = index.to_summary(max_tokens=budget)
        assert _summary_tokens(summary) <= budget
        shape = tuple(summary) + tuple(summary.get("omitted", ())) + tuple(summary.get("stats", ()))
        if not shapes or shapes[-1] != shape:
            shapes.append(shape)
        if budget >= 6:
            assert summary["truncated"] is True
        if "omitted" in summary:
            assert all(summary["omitted"].values())  # only what there is to report
            assert summary["omitted"]["files"] == 60
    # each step keeps strictly more than the one before it
    assert shapes[0] == () and shapes[1] == ("truncated",)
    assert shapes[2] == ("truncated", "omitted", "files")
    assert all(len(a) < len(b) for a, b in zip(shapes, shapes[1:], strict=False))
    richest = index.to_summary(max_tokens=full_at - 1)
    assert set(richest) == {"stats", "truncated", "omitted"}
    assert richest["stats"] == index.to_summary(max_tokens=full_at)["stats"]
    assert richest["omitted"]["symbols"] > 0
    # `{}` (one token) is the smallest JSON object, so max_tokens=0 cannot be met
    assert index.to_summary(max_tokens=0) == {} and _summary_tokens({}) == 1


# ---- PHP -------------------------------------------------------------------------------


def test_r2_php_slim_map_any_and_fastroute_addroute_are_routes(tmp_path: Path) -> None:
    rec = _record(tmp_path, "routes.php", (
        "<?php\n"
        "$app->map(['GET', 'POST'], '/slim/both', 'handler');\n"
        "$app->any('/slimany', 'h');\n"
        "$router->addRoute('GET', '/fast', 'handler');\n"
        "$r->addRoute(['PUT', 'PATCH'], '/fast/{id}', 'update_handler');\n"
        "$app->get('/slim/plain', 'list_things');\n"
        "$app->any('/slim/var', $handler);\n"
        # the same call shapes that are not routes
        "$app->get('setting', 'fallback');\n"       # container / config lookup
        "$api->post('/charges', $payload);\n"       # HTTP client
        "$r->get('page', 1);\n"                     # request accessor
        "$router->addRoute($method, '/dynamic', 'h');\n"
        "$router->addRoute('label', '/not-a-method', 'h');\n"
        "$items->map(fn ($x) => $x + 1);\n"
    ))
    assert [(s.name, s.meta["methods"], s.meta["handler"]) for s in rec.symbols] == [
        ("GET|POST /slim/both", ["GET", "POST"], "handler"),
        ("ANY /slimany", ["ANY"], "h"),
        ("GET /fast", ["GET"], "handler"),
        ("PUT|PATCH /fast/{id}", ["PUT", "PATCH"], "update_handler"),
        ("GET /slim/plain", ["GET"], "list_things"),
        ("ANY /slim/var", ["ANY"], "$handler"),
    ]
    assert len(_edges(rec, "route")) == 6
    assert {"->map", "->any", "->addRoute"} <= _targets(rec, "call")  # still call edges


def test_r2_php_use_const_resolves_to_the_declaring_file(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "src/Support/helpers.php": (
            "<?php\nnamespace App\\Support;\n\nconst MAX_ROWS = 100;\n"
            "function helper_fn(): int { return MAX_ROWS; }\ndefine('LEGACY_FLAG', true);\n"
            "final class Limits { const MAX_ROWS = 5; }\n"
        ),
        "src/Other/a.php": "<?php\nnamespace App\\Other;\nconst DUP = 1;\n",
        "src/Other/b.php": "<?php\nnamespace App\\Other;\nconst DUP = 2;\n",
        "src/use.php": (
            "<?php\nnamespace App;\n\n"
            "use const App\\Support\\MAX_ROWS;\n"
            "use const app\\support\\MAX_ROWS as LIMIT;\n"   # namespaces are case-insensitive
            "use const App\\Support\\max_rows;\n"            # constant names are not
            "use function App\\Support\\helper_fn;\n"
            "use const LEGACY_FLAG;\n"
            "use const App\\Other\\DUP;\n"                   # declared twice: ambiguous
            "use const App\\Support\\{MAX_ROWS as GROUPED, MISSING};\n"
        ),
    }))
    use = index.file("src/use.php")
    assert use is not None
    helpers = "src/Support/helpers.php"
    assert [(e.line, e.target, e.resolved_file, e.resolved_symbol)
            for e in _edges(use, "import")] == [
        (4, "App\\Support\\MAX_ROWS", helpers, "App\\Support\\MAX_ROWS"),
        (5, "app\\support\\MAX_ROWS", helpers, "App\\Support\\MAX_ROWS"),
        (6, "App\\Support\\max_rows", None, None),
        (7, "App\\Support\\helper_fn", helpers, "App\\Support\\helper_fn"),
        (8, "LEGACY_FLAG", helpers, "LEGACY_FLAG"),
        (9, "App\\Other\\DUP", None, None),
        (10, "App\\Support\\MAX_ROWS", helpers, "App\\Support\\MAX_ROWS"),
        (10, "App\\Support\\MISSING", None, None),
    ]
    assert helpers in index.dependencies_of("src/use.php")


# ---- JavaScript / TypeScript -----------------------------------------------------------


def test_r2_js_api_calls_request_objects_jquery_options_and_client_labels(
    tmp_path: Path,
) -> None:
    rec = _record(tmp_path, "web/api.js", (
        "import axios from 'axios';\n"
        "const client = axios.create({ baseURL: '/api/v2' });\n"
        "function calls(method, id, other) {\n"
        "  fetch(new Request('/d', { method: 'PATCH' }));\n"
        "  fetch(new Request('/d2'));\n"
        "  fetch(new Request('/d3', { method: 'PUT' }), { method: 'DELETE' });\n"
        "  $.post({ url: '/jq5' });\n"
        "  $.get({ url: '/jq6', data: {} });\n"
        "  const x = new XMLHttpRequest();\n"
        "  x.open(method, '/xhr1');\n"
        "  x.open('POST', '/xhr2');\n"
        "  http.get('/not-axios');\n"
        "  request.get('/superagent');\n"
        "  this.http.post('/angular');\n"
        "  axios.get('/real-axios');\n"
        "  client.get('/inst');\n"
        "  window.open(url, '_blank');\n"       # not XHR: literal first argument, no verb
        "  other.open(method, '/not-xhr');\n"   # not XHR: unknown receiver
        "}\n"
    ))
    assert [(e.line, e.target, e.meta["client"]) for e in _edges(rec, "api_call")] == [
        (4, "PATCH /d", "fetch"),
        (5, "GET /d2", "fetch"),
        (6, "DELETE /d3", "fetch"),  # the init passed to fetch() overrides the Request's
        (7, "POST /jq5", "jquery"),
        (8, "GET /jq6", "jquery"),
        (10, "ANY /xhr1", "xhr"),  # the method is a variable
        (11, "POST /xhr2", "xhr"),
        (12, "GET /not-axios", "http"),
        (13, "GET /superagent", "http"),
        (14, "POST /angular", "http"),
        (15, "GET /real-axios", "axios"),
        (16, "GET /api/v2/inst", "axios"),  # an axios.create() instance
    ]
    typed = _record(tmp_path, "web/typed.ts", (
        "export function send(req: XMLHttpRequest, verb: string, xhrLike: any) {\n"
        "  req.open(verb, `/typed/${verb}`);\n  xhrLike.open(verb, '/named');\n}\n"
    ))
    assert _targets(typed, "api_call") == {"ANY /typed/{verb}", "ANY /named"}


def test_r2_js_handlers_skip_detached_values_and_event_plumbing(tmp_path: Path) -> None:
    rec = _record(tmp_path, "web/Form.jsx", (
        "export function Form({ save }) {\n"
        "  el.onclick = null;\n"
        "  el.onchange = undefined;\n"
        "  el.onkeyup = void 0;\n"
        "  form.onsubmit = async (e) => { e.preventDefault(); await save(); };\n"
        "  btn.onclick = (e) => { e.stopPropagation(); console.log(e); };\n"
        "  btn.addEventListener('click', (e) => { e.preventDefault(); submit(e); });\n"
        "  btn.addEventListener('click', null);\n"
        "  return <Fancy onClick={undefined} onChange={null} onFocus={handleFocus}\n"
        "    onBlur={(e) => { e.preventDefault(); track('blur'); }} />;\n"
        "}\n"
    ))
    assert [(e.line, e.meta["event"], e.target) for e in _edges(rec, "handler")] == [
        (5, "submit", "save"),
        (6, "click", "<inline>"),  # nothing but plumbing in the body
        (7, "click", "submit"),
        (9, "onFocus", "handleFocus"),
        (10, "onBlur", "track"),
    ]
    assert {e.source for e in _edges(rec, "handler")} == {"Form"}


def test_r2_react_hooks_called_through_a_wrapper_belong_to_the_component(
    tmp_path: Path,
) -> None:
    rec = _record(tmp_path, "web/UserRow.jsx", (
        "export const UserRow = ({ id }) => {\n"
        "  const onDelete = useCallback(() => removeUser(id), [id]);\n"
        "  const [open, setOpen] = useState(false);\n"
        "  return <tr onClick={onDelete}><td>{id}</td></tr>;\n"
        "};\n"
        "export function useThing() {\n"
        "  const cb = useCallback(function () { return useInner(); }, []);\n"
        "  return cb;\n"
        "}\n"
        "export const Card = memo(() => <b />);\n"
    ))
    row = _sym(rec, "UserRow", "component")
    assert row.meta["hooks"] == ["useCallback", "useState"]
    assert _targets(rec, "uses_hook", "UserRow") == {"useCallback", "useState"}
    callback = _sym(rec, "UserRow.onDelete", "function")
    assert callback.meta == {"wrapper": "useCallback"}  # the callback itself uses no hook
    assert _targets(rec, "call", "UserRow.onDelete") == {"removeUser"}
    assert "useCallback" in _targets(rec, "call", "UserRow")
    assert _sym(rec, "useThing", "hook").meta["hooks"] == ["useCallback"]
    # a hook called inside the wrapped function still belongs to that function
    assert _targets(rec, "uses_hook", "useThing.cb") == {"useInner"}
    # non-hook wrappers keep their call on the symbol they define
    assert _targets(rec, "call", "Card") == {"memo"}


def test_r2_js_function_expression_signatures_keep_star_name_and_generics(
    tmp_path: Path,
) -> None:
    rec = _record(tmp_path, "web/sigs.tsx", (
        "const three = function* () { yield 1; };\n"
        "let f = async function* named() { yield 2; };\n"
        "export const Inner2 = React.memo(function Inner(props: Props) { return <b/>; });\n"
        "export const Inp = forwardRef<HTMLInputElement, Props>((props, ref) => {\n"
        "  return <input ref={ref} />;\n});\n"
        "const id = <T,>(x: T): T => x;\n"
        "const plain = function () { return 1; };\n"
        "class K {\n"
        "  123() { return 1; }\n"
        "  gen = function* inner() {};\n"
        "  plain = function () { return 1; };\n"
        "}\n"
    ))
    assert {s.qualified_name: (s.signature, s.modifiers) for s in rec.symbols} == {
        "three": ("const three = function* ()", ("generator", "function-expression")),
        "f": ("let f = async function* named()", ("async", "generator", "function-expression")),
        "Inner2": ("export const Inner2 = React.memo(function Inner(props: Props)",
                   ("export", "function-expression")),
        "Inp": ("export const Inp = forwardRef<HTMLInputElement, Props>((props, ref) =>",
                ("export", "arrow")),
        "id": ("const id = <T,>(x: T): T =>", ("arrow",)),
        "plain": ("const plain = function ()", ("function-expression",)),
        "K": ("class K", ()),
        "K.123": ("123()", ()),
        "K.gen": ("gen = function* inner()", ("function-expression", "generator")),
        "K.plain": ("plain = function ()", ("function-expression",)),
    }
    assert _sym(rec, "Inner2").meta["wrapper"] == "memo"  # the meta stays the bare name
    assert (_sym(rec, "Inner2").kind, _sym(rec, "K.123").kind) == ("component", "method")
    # a pathological name / generic list cannot grow the signature past its limit
    long = _record(tmp_path, "web/long.js", (
        f"const x = function* {'n' * 5000}() {{}};\n"
        f"const y = memo<{'T, ' * 3000}U>(() => 1);\n"
    ))
    assert all(len(s.signature) <= MAX_SIGNATURE_CHARS for s in long.symbols)


# ---- Cross-file resolution -------------------------------------------------------------


def test_r2_inferred_url_to_route_matches_are_all_labelled(tmp_path: Path) -> None:
    index = _index(_write(tmp_path, {
        "routes/api.php": (
            "<?php\nRoute::get('/users/{id}', [UserController::class, 'show']);\n"
            "Route::get('/api/v1/things/{id}', [ThingController::class, 'show']);\n"
            "Route::get('/api/users', [UserController::class, 'index']);\n"
        ),
        "web/a.js": (
            "const API = '/api';\n"
            "function f(id) {\n"
            "  fetch('users/5');\n"
            "  fetch(BASE + '/api/v1/things/' + id);\n"
            "  fetch(`${BASE_URL}/users/${id}`);\n"
            "  fetch('http://localhost:8000/api/v1/things/9');\n"
            "  fetch('https://other.example.com/api/users');\n"
            "  fetch('/v1/things/2');\n"
            "  fetch('/users/7');\n"
            "  fetch(`${API}/users`);\n"
            "  fetch('/nope');\n"
            "}\n"
        ),
    }))
    client = index.file("web/a.js")
    assert client is not None
    assert [(e.target, e.resolved_symbol, e.meta.get("resolution"))
            for e in _edges(client, "api_call")] == [
        ("GET users/5", "GET /users/{id}", "relative-url"),
        ("GET {BASE}/api/v1/things/{id}", "GET /api/v1/things/{id}", "base-placeholder"),
        ("GET {BASE_URL}/users/{id}", "GET /users/{id}", "base-placeholder"),
        ("GET http://localhost:8000/api/v1/things/9", "GET /api/v1/things/{id}",
         "local-origin"),
        ("GET https://other.example.com/api/users", "GET /api/users", "origin-ignored"),
        ("GET /v1/things/2", "GET /api/v1/things/{id}", "suffix"),
        ("GET /users/7", "GET /users/{id}", None),  # exact: nothing was assumed
        ("GET /api/users", "GET /api/users", None),  # a known constant is not a placeholder
        ("GET /nope", None, None),  # and an unresolved edge never carries a label
    ]
    calls = {c["call"]: c.get("match") for c in index.to_summary(8000)["api_calls"]}
    assert calls["GET users/5"] == "relative-url"
    assert calls["GET {BASE_URL}/users/{id}"] == "base-placeholder"
    assert calls["GET /users/7"] is None


# ---- Optional tree-sitter backend ------------------------------------------------------


def test_r2_tree_sitter_type_literal_members_are_not_methods(tmp_path: Path) -> None:
    """`type T = { m(): void }` yielded a `method T.m` symbol only with tree-sitter."""
    body = (
        "type T1 = { m(): void; f: () => void };\n"
        "interface I1 { m(): void; g(x: number): string }\n"
        "export type T2 = {\n  run(): void;\n};\n"
        "function f(x: { m(): void }): { n(): void } { return x as any; }\n"
        "class K { p: { q(): void }; r(): void {} }\n"
    )
    with_ts = _tree_sitter_indexer(tmp_path).index_source("a.ts", body)
    regex = _indexer(tmp_path).index_source("a.ts", body)
    assert with_ts.extractor == "typescript-regex+tree-sitter"
    assert [s.to_dict() for s in with_ts.symbols] == [s.to_dict() for s in regex.symbols]
    assert _names(regex, "method") == ["I1.m", "I1.g", "K.r"]
    # the round-2 signature and semantic changes are identical in both modes
    sigs = (
        "const three = function* () { yield 1; };\n"
        "export const Inp = forwardRef<HTMLInputElement, Props>((props, ref) => {\n"
        "  const cb = useCallback(() => props.go(ref), []);\n  el.onclick = null;\n"
        "  fetch(new Request('/d', { method: 'PATCH' }));\n  return <input ref={ref} />;\n});\n"
    )
    with_ts = _tree_sitter_indexer(tmp_path).index_source("b.tsx", sigs)
    regex = _indexer(tmp_path).index_source("b.tsx", sigs)
    assert with_ts.extractor == "typescript-regex+tree-sitter"
    assert [s.to_dict() for s in with_ts.symbols] == [s.to_dict() for s in regex.symbols]
    assert [e.to_dict() for e in with_ts.edges] == [e.to_dict() for e in regex.edges]


# ======================================================================================
# Gate 15: Review Round 3 Regressions (one focused test per residual gap)
# ======================================================================================
# The medium gap (unclosed `<` 50-350x slower than valid code) is covered in Gate 11 by
# test_unclosed_angle_brackets_cost_about_what_valid_code_costs,
# test_generic_list_table_matches_a_forward_scan and the `unclosed_*` pathological cases.

# ---- PHP -------------------------------------------------------------------------------


def test_r3_php_symfony_methods_mix_constants_and_string_literals(tmp_path: Path) -> None:
    """`methods: [Request::METHOD_POST, 'PUT']` used to lose the constant (-> `PUT /a`)."""
    rec = _record(tmp_path, "src/Controller/MixController.php", (
        "<?php\nnamespace App\\Controller;\n"
        "use Symfony\\Component\\HttpFoundation\\Request;\n"
        "class MixController {\n"
        "    #[Route('/a', methods: [Request::METHOD_POST, 'PUT'])]\n"
        "    public function a() {}\n"
        "    #[Route('/c', methods: ['put', Request::METHOD_POST])]\n"
        "    public function c() {}\n"
        "    #[Route('/d', methods: [Request::METHOD_GET, Request::METHOD_DELETE])]\n"
        "    public function d() {}\n"
        "    #[Route('/e', methods: ['GET', 'HEAD'])]\n"
        "    public function e() {}\n"
        "    #[Route('/f', methods: ['GET', Request::METHOD_GET, 'POST'])]\n"
        "    public function f() {}\n"
        "}\n"
    ))
    assert {s.name: s.meta["methods"] for s in rec.symbols if s.kind == "route"} == {
        "POST|PUT /a": ["POST", "PUT"],  # source order, whichever kind comes first
        "PUT|POST /c": ["PUT", "POST"],
        "GET|DELETE /d": ["GET", "DELETE"],
        "GET|HEAD /e": ["GET", "HEAD"],
        "GET|POST /f": ["GET", "POST"],  # the same verb named twice is one verb
    }
    laravel = _record(tmp_path, "routes/web.php", (
        "<?php\nRoute::match([Request::METHOD_GET, 'post'], '/m', [MixController::class, 'm']);\n"
    ))
    assert _names(laravel, "route") == ["GET|POST /m"]


def test_r3_php_laravel_resource_update_redirects_and_group_namespaces(tmp_path: Path) -> None:
    """The resource update action answers PUT and PATCH; `Route::redirect` is a route; an
    array / fluent group `namespace` applies to string handlers (and only to those)."""
    index = _index(_write(tmp_path, {
        "routes/web.php": (
            "<?php\nuse App\\Http\\Controllers\\PhotoController;\n"
            "Route::resource('photos', PhotoController::class);\n"
            "Route::apiResource('tags', PhotoController::class)->only(['update']);\n"
            "Route::redirect('/here', '/there');\n"
            "Route::permanentRedirect('/old', '/new');\n"
            "Route::group(['namespace' => 'Admin', 'prefix' => 'admin'], function () {\n"
            "    Route::get('/users', 'UserController@index');\n"
            "    Route::get('/abs', '\\App\\Http\\Controllers\\PhotoController@index');\n"
            "    Route::get('/cls', [PhotoController::class, 'index']);\n"
            "    Route::resource('posts', 'PostController')->only(['index']);\n"
            "});\n"
            "Route::namespace('Api')->group(function () {\n"
            "    Route::get('/api/users', 'UserController@index');\n"
            "});\n"
            "Route::get('/plain', 'UserController@index');\n"
        ),
        "app/Http/Controllers/PhotoController.php": (
            "<?php\nnamespace App\\Http\\Controllers;\n"
            "class PhotoController { public function index() {} public function update() {} }\n"
        ),
        "app/Http/Controllers/Admin/UserController.php": (
            "<?php\nnamespace App\\Http\\Controllers\\Admin;\n"
            "class UserController { public function index() {} }\n"
        ),
        "app/Http/Controllers/Api/UserController.php": (
            "<?php\nnamespace App\\Http\\Controllers\\Api;\n"
            "class UserController { public function index() {} }\n"
        ),
        "web/app.js": (
            "fetch('/photos/5', {method: 'PATCH'});\n"
            "fetch('/photos/5', {method: 'PUT'});\n"
            "fetch('/photos/5', {method: 'DELETE'});\n"
            "fetch('/here');\n"
        ),
    }))
    routes = index.file("routes/web.php")
    client = index.file("web/app.js")
    assert routes is not None and client is not None
    handlers = {s.name: s.meta["handler"] for s in routes.symbols if s.kind == "route"}
    ctl = "App\\Http\\Controllers\\PhotoController"
    assert handlers["PUT|PATCH /photos/{id}"] == ctl + "::update"
    assert _sym(routes, "PUT|PATCH /photos/{id}").meta["methods"] == ["PUT", "PATCH"]
    assert "PUT /photos/{id}" not in handlers and handlers["PUT|PATCH /tags/{id}"] == ctl + "::update"
    assert handlers["ANY /here"] == "redirect:/there" and handlers["ANY /old"] == "redirect:/new"
    assert handlers["GET /admin/users"] == "Admin\\UserController::index"
    assert handlers["GET /admin/abs"] == ctl + "::index"  # written fully qualified
    assert handlers["GET /admin/cls"] == ctl + "::index"  # ::class needs no namespace
    assert handlers["GET /admin/posts"] == "Admin\\PostController::index"
    assert handlers["GET /api/users"] == "Api\\UserController::index"
    assert handlers["GET /plain"] == "UserController::index"
    # both verbs reach the update route; the redirect has no handler edge to chase
    assert {e.target: e.resolved_symbol for e in _edges(client, "api_call")} == {
        "PATCH /photos/5": "PUT|PATCH /photos/{id}",
        "PUT /photos/5": "PUT|PATCH /photos/{id}",
        "DELETE /photos/5": "DELETE /photos/{id}",
        "GET /here": "ANY /here",
    }
    assert not any(e.target.startswith("redirect:") for e in _edges(routes, "route"))
    # the namespace tells the two UserControllers apart: resolved, and labelled as inferred
    # because the root namespace the framework prepends is not in the source
    edges = {e.source: e for e in _edges(routes, "route")}
    admin, api, plain = edges["GET /admin/users"], edges["GET /api/users"], edges["GET /plain"]
    assert (admin.resolved_file, admin.resolved_symbol, admin.meta["resolution"]) == (
        "app/Http/Controllers/Admin/UserController.php",
        "App\\Http\\Controllers\\Admin\\UserController::index", "suffix",
    )
    assert api.resolved_file == "app/Http/Controllers/Api/UserController.php" and api.inferred
    assert plain.resolved_file is None  # two candidates and no namespace: never a guess
    assert edges["GET /admin/posts"].resolved_file is None  # no such class anywhere


def test_r3_php_every_declarator_of_a_property_list_is_indexed(tmp_path: Path) -> None:
    """`public $a, $b = [1, 2], $c;` used to index `$a` only."""
    rec = _record(tmp_path, "src/Bag.php", (
        "<?php\nclass Bag {\n"
        "    private static array $registry = [], $other = [];\n"
        "    public $a, $b = [1, 2], $c;\n"
        "    public int $x = 1,\n"
        "               $y = 2;\n"
        "    protected ?Foo $single;\n"
        "    public function __construct(private int $pa, $pb, public string $pc = 'x') {}\n"
        "    public function m($p, $q) { static $s1, $s2; $t = [$p, $q]; }\n"
        "}\n"
    ))
    props = {s.name: s for s in rec.symbols if s.kind == "property"}
    assert list(props) == ["$registry", "$other", "$a", "$b", "$c", "$x", "$y", "$single",
                           "$pa", "$pc"]
    assert props["$other"].signature == "private static array $other"
    assert props["$other"].modifiers == ("private", "static")
    assert props["$other"].qualified_name == "Bag::$other" and props["$other"].parent == "Bag"
    assert (props["$b"].signature, props["$c"].signature) == ("public $b", "public $c")
    assert (props["$x"].line, props["$y"].line) == (5, 6)  # each on its own line
    assert props["$y"].signature == "public int $y"
    # plain constructor parameters, locals and `static` locals are not properties
    assert not {"$pb", "$p", "$q", "$s1", "$s2", "$t"} & set(props)


# ---- JavaScript / TypeScript -----------------------------------------------------------


def test_r3_ts_type_literal_member_signatures_are_not_calls(tmp_path: Path) -> None:
    """`type Obj = { b(): void; c(x: number): string }` reported calls to `b` and `c`."""
    rec = _record(tmp_path, "web/types.ts", (
        "export type Obj = { a: string; b(): void; c(x: number): string };\n"
        "type Multi = {\n"
        "  first: string\n"
        "  second(): void\n"
        "  third<T>(x: T): T\n"
        "  fourth(): Promise<void>,\n"
        "}\n"
        "function f(opts: { cb(): void }): { done(): boolean } {\n"
        "  const o = { key: test ? yes(1) : no(2), other: fn(3), m(): number { return g(4); } };\n"
        "  switch (v) { case k(1): return opts; }\n"
        "  return cond ? h(5) : j(6);\n"
        "}\n"
        "const pick = flag\n"
        "  ? first(7)\n"
        "  : second(8);\n"
        "interface I { m(): void; n(x: number): string }\n"
    ))
    assert _targets(rec, "call") == {
        "yes", "no", "fn", "g", "k", "h", "j", "first", "second",
    }
    assert _targets(rec, "call", "web/types.ts") == {"first", "second"}  # nothing from a type
    assert _names(rec, "type_alias") == ["Obj", "Multi"]
    assert _names(rec, "method") == ["I.m", "I.n"]  # interface members stay signatures
    # plain JavaScript has no type literals: the same text there is left alone
    js = _record(tmp_path, "web/plain.js", "const v = c ? f(1) : g(2);\nh(3);\n")
    assert _targets(js, "call") == {"f", "g", "h"}


def test_r3_js_statement_ends_are_memoized_without_changing_declarators(
    tmp_path: Path,
) -> None:
    """Every crossed token is memoized now (not only line breaks) and an unclosed `Name<`
    no longer hides the commas after it; declarator lists must come out the same."""
    rec = _record(tmp_path, "web/decl.ts", (
        "let p = i<n, q = 2;\n"
        "let m: Map<string, number> = new Map(), z = 1;\n"
        "const f = (a) => a<b\n"
        "foo()\n"
        "let w = x<y\n"
        "bar(w>1)\n"
        "let one = f1() let two = f2() let three = f3()\n"
        "const multi = first +\n"
        "  second,\n"
        "  last = 3;\n"
    ))
    assert _names(rec, "variable") == ["p", "q", "m", "z", "w", "one", "two", "three"]
    assert _names(rec, "constant") == ["multi", "last"]
    assert _sym(rec, "m").signature == "let m: Map<string, number>"
    assert _sym(rec, "f", "function").end_line == 3  # the arrow body ends with its line
    assert _targets(rec, "call", "web/decl.ts") == {"Map", "foo", "bar", "f1", "f2", "f3"}


# ---- Lexers ----------------------------------------------------------------------------


def test_r3_unterminated_block_comments_are_recorded(tmp_path: Path) -> None:
    """An open `/*` silently masked the rest of the file: `errors` stayed empty."""
    indexer = _indexer(tmp_path)
    tail = "is never closed; the rest of the file was treated as its content"
    js = indexer.index_source("c.js", "function ok() {}\n/* never closed\nfunction lost() {}\n")
    assert _names(js, "function") == ["ok"]
    assert js.errors == (f"javascript: the block comment opened on line 2 {tail}",)
    ts = indexer.index_source("c.tsx", "\n\n\nexport const a = 1; /** doc\n * b\n")
    assert ts.errors == (f"typescript: the block comment opened on line 4 {tail}",)
    php = indexer.index_source("c.php", "<?php\nfunction ok() {}\n/* never closed\nfunction lost() {}\n")
    assert _names(php, "function") == ["ok"]
    assert php.errors == (f"php: the block comment opened on line 3 {tail}",)
    sql = indexer.index_source(
        "c.sql", "CREATE TABLE ok (id int);\n/* outer /* nested */ never closed\nCREATE TABLE lost (id int);\n"
    )
    assert _names(sql, "table") == ["ok"]
    assert sql.errors == (
        "sql: unterminated block comment at line 2; the rest of the file was treated as "
        "its content",
    )
    # closed comments (also nested SQL ones and `/*` inside strings) are not reported
    for rel, body in (
        ("ok.js", "/* a */ function f() {} // b /* c\nconst s = '/* not a comment';\n"),
        ("ok.php", "<?php /* a */ function f() {} # b /* c\n$s = '/* not a comment';\n"),
        ("ok.sql", "/* a /* nested */ b */ CREATE TABLE t (id int); -- c /* d\n"),
    ):
        assert indexer.index_source(rel, body).errors == (), rel
    # the error survives serialization and an incremental run re-reads nothing
    _write(tmp_path, {"c.js": "function ok() {}\n/* never closed\n"})
    first = indexer.index()
    again = indexer.index(previous=first)
    assert again.to_json() == first.to_json() and indexer.last_run.files_reused == 1
    assert RepoIndex.from_json(first.to_json()).files[0].errors == first.files[0].errors


# ---- Cross-file resolution -------------------------------------------------------------


def test_r3_catch_all_route_parameters_match_multi_segment_urls(tmp_path: Path) -> None:
    """Flask `<path:rest>` (and its FastAPI / Slim spellings) swallows the rest of the URL."""
    index = _index(_write(tmp_path, {
        "app.py": (
            "from flask import Flask\napp = Flask(__name__)\n\n"
            "@app.get('/api/files/<path:rest>')\ndef files(rest): ...\n\n"
            "@app.get('/api/files/meta/info')\ndef info(): ...\n\n"
            "@app.get('/api/docs/<int:page>')\ndef docs(page): ...\n\n"
            "@app.get('/static/{rest:path}')\ndef static(rest): ...\n\n"
            "@app.post('/upload/<path:rest>')\ndef upload(rest): ...\n"
        ),
        "routes.php": "<?php\n$app->get('/blob/{rest:.*}', [BlobController::class, 'show']);\n",
        "web/a.js": (
            "fetch('/api/files/a/b/c');\n"
            "fetch('/api/files/one');\n"
            "fetch('/api/files/meta/info');\n"
            "fetch(`/api/files/${dir}/${name}`);\n"
            "fetch('/api/docs/1/2');\n"
            "fetch('/static/css/site/main.css');\n"
            "fetch('/blob/x/y');\n"
            "fetch('/upload/a/b');\n"
            "fetch('/api/files');\n"
        ),
    }))
    client = index.file("web/a.js")
    assert client is not None
    assert [(e.target, e.resolved_symbol, e.meta.get("resolution"))
            for e in _edges(client, "api_call")] == [
        ("GET /api/files/a/b/c", "GET /api/files/<path:rest>", None),
        ("GET /api/files/one", "GET /api/files/<path:rest>", None),
        ("GET /api/files/meta/info", "GET /api/files/meta/info", None),  # spelled out wins
        ("GET /api/files/{dir}/{name}", "GET /api/files/<path:rest>", None),
        ("GET /api/docs/1/2", None, None),  # an ordinary parameter is one segment
        ("GET /static/css/site/main.css", "GET /static/{rest:path}", None),
        ("GET /blob/x/y", "GET /blob/{rest:.*}", None),
        ("GET /upload/a/b", None, None),  # the method still has to agree
        ("GET /api/files", None, None),  # a catch-all needs something to catch
    ]


def test_r3_inferred_resolutions_can_be_told_apart_from_exact_ones(tmp_path: Path) -> None:
    """Inferred matches set `resolved_file` like exact ones; consumers had to read
    `meta["resolution"]` themselves to separate them in dependency lists."""
    index = _index(_write(tmp_path, {
        "app.py": (
            "from flask import Flask\napp = Flask(__name__)\n\n"
            "@app.get('/api/users/me')\ndef me(): ...\n\n"
            "@app.route('/py/multi', methods=['GET', 'PUT'])\ndef multi(): ...\n"
        ),
        "web/guess.js": (
            "fetch('https://api.github.com/api/users/me');\n"
            "fetch('/multi', {method: cond ? 'POST' : 'PUT'});\n"
        ),
        "web/exact.js": "fetch('/api/users/me');\n",
        "web/both.js": "fetch('/api/users/me');\nfetch('/multi');\n",
    }))
    guess = index.file("web/guess.js")
    assert guess is not None
    assert [(e.meta.get("resolution"), e.inferred) for e in _edges(guess, "api_call")] == [
        ("origin-ignored", True), ("suffix", True),
    ]
    exact_edge = _edges(index.file("web/exact.js"), "api_call")[0]  # type: ignore[arg-type]
    assert exact_edge.resolved_file == "app.py" and exact_edge.inferred is False
    assert Edge("call", "a", "b", "f.js", 1, None, None, {"resolution": "suffix"}).inferred is False
    # the default is unchanged (every resolved edge counts) ...
    for path in ("web/guess.js", "web/exact.js", "web/both.js"):
        assert index.dependencies_of(path) == ["app.py"]
    assert index.dependents_of("app.py") == ["web/both.js", "web/exact.js", "web/guess.js"]
    # ... and the exact ones can be asked for
    assert index.dependencies_of("web/guess.js", include_inferred=False) == []
    assert index.dependencies_of("web/both.js", include_inferred=False) == ["app.py"]
    assert index.inferred_dependencies_of("web/guess.js") == ["app.py"]
    assert index.inferred_dependencies_of("web/both.js") == []  # one exact edge is enough
    assert index.inferred_dependencies_of("web/missing.js") == []
    assert index.dependents_of("app.py", include_inferred=False) == [
        "web/both.js", "web/exact.js",
    ]
    assert index.dependents_of("app.py") == ["web/both.js", "web/exact.js", "web/guess.js"]
    # the LLM-facing views say so too, inside their budgets
    files = {entry["path"]: entry for entry in index.to_summary()["files"]}
    assert files["web/guess.js"]["deps"] == files["web/guess.js"]["deps_inferred"] == ["app.py"]
    assert files["web/both.js"]["deps"] == ["app.py"] and "deps_inferred" not in files["web/both.js"]
    assert "deps_inferred" not in files["web/exact.js"]
    full = index.render_outline(max_tokens=10_000)
    assert "GET https://api.github.com/api/users/me -> app.py::GET /api/users/me (origin-ignored)" in full
    assert "ANY /multi -> app.py::GET|PUT /py/multi (suffix)" in full
    assert "api: GET /api/users/me -> app.py::GET /api/users/me\n" in full  # exact: unmarked
    outlines = [index.render_outline(max_tokens=budget) for budget in range(0, 400)]
    assert any("  deps: app.py (inferred)" in text for text in outlines)
    assert any("web/exact.js [javascript 1L]\n  deps: app.py\n" in text for text in outlines)
    for budget in range(0, 400):
        assert estimate_tokens(outlines[budget]) <= budget
        assert _summary_tokens(index.to_summary(max_tokens=budget)) <= max(budget, 1)
    # the labels round-trip, so a loaded index answers the same
    loaded = RepoIndex.from_json(index.to_json())
    assert loaded.dependencies_of("web/guess.js", include_inferred=False) == []
    assert loaded.inferred_dependencies_of("web/guess.js") == ["app.py"]


def test_r3_python_relative_imports_in_a_src_layout_use_importable_names(
    tmp_path: Path,
) -> None:
    """`from . import b` in src/proj/a.py was named `src.proj.b` while `from proj import b`
    was `proj.b`: one module, two names."""
    files = {
        "src/proj/__init__.py": "from .a import run\n",
        "src/proj/a.py": (
            "from . import b\nfrom .sub.deep import fn\nfrom proj import b as b2\n"
            "from .. import beyond\ndef run(): ...\n"
        ),
        "src/proj/b.py": "x = 1\n",
        "src/proj/sub/__init__.py": "",
        "src/proj/sub/deep.py": "from ..b import x\ndef fn(): ...\n",
        "app/__init__.py": "",
        "app/m.py": "from . import n\n",
        "app/n.py": "",
        "tools/pkg/__init__.py": "",
        "tools/pkg/a.py": "from . import b\n",
        "tools/pkg/b.py": "",
        "scripts/x.py": "from . import y\n",
        "scripts/y.py": "",
    }
    indexer = _indexer(_write(tmp_path, files))
    index = indexer.index()

    def imports(path: str) -> list[tuple[str, str | None, str | None, dict[str, Any]]]:
        rec = index.file(path)
        assert rec is not None
        return [(e.target, e.resolved_file, e.resolved_symbol, dict(e.meta))
                for e in _edges(rec, "import")]

    assert imports("src/proj/a.py") == [
        ("proj.b", "src/proj/b.py", None,
         {"level": 1, "module": "proj", "name": "b", "root": "src"}),
        ("proj.sub.deep.fn", "src/proj/sub/deep.py", "fn",
         {"level": 1, "module": "proj.sub.deep", "name": "fn", "root": "src"}),
        # the absolute spelling of the same module: same name (resolved by inference)
        ("proj.b", "src/proj/b.py", None,
         {"alias": "b2", "module": "proj", "name": "b", "resolution": "suffix"}),
        # climbing out of the top-level package is left exactly as the path spells it
        ("src.beyond", None, None, {"level": 2, "module": "src", "name": "beyond"}),
    ]
    assert imports("src/proj/sub/deep.py") == [
        ("proj.b.x", "src/proj/b.py", "x",
         {"level": 2, "module": "proj.b", "name": "x", "root": "src"}),
    ]
    assert imports("src/proj/__init__.py")[0][:3] == ("proj.a.run", "src/proj/a.py", "run")
    assert not any(e.inferred for e in _edges(index.file("src/proj/sub/deep.py"), "import"))  # type: ignore[arg-type]
    # only a conventional source root above a regular package is stripped
    assert imports("app/m.py")[0][0] == "app.n"
    assert imports("tools/pkg/a.py")[0][0] == "tools.pkg.b"
    assert imports("scripts/x.py")[0][0] == "scripts.y"
    assert all("root" not in meta for path in ("app/m.py", "tools/pkg/a.py", "scripts/x.py")
               for _, _, _, meta in imports(path))
    # one file on its own cannot know that src/ is not a package
    assert _edges(indexer.index_file("src/proj/a.py"), "import")[0].target == "src.proj.b"
    # a reused record resolves to the very same bytes, and names follow the tree: once
    # src/ becomes a package the path-based names are back, and they flip again without it
    again = indexer.index(previous=index)
    assert again.to_json() == index.to_json() and indexer.last_run.files_parsed == 0
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    packaged = indexer.index(previous=index)
    rec = packaged.file("src/proj/a.py")
    assert rec is not None
    assert [(e.target, e.resolved_file, e.meta.get("root")) for e in _edges(rec, "import")] == [
        ("src.proj.b", "src/proj/b.py", None),
        ("src.proj.sub.deep.fn", "src/proj/sub/deep.py", None),
        ("proj.b", None, None),  # `proj` is not importable when src is the package
        ("src.beyond", "src/__init__.py", None),  # ... and `..` now names that package
    ]
    assert packaged.to_json() == _indexer(tmp_path).index().to_json()
    (tmp_path / "src" / "__init__.py").unlink()
    assert indexer.index(previous=packaged).to_json() == index.to_json()
