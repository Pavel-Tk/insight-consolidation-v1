"""Surface realization: turning a latent schema plus a seed into a pile of business documents.

Everything here is deterministic given a seed, so an episode is fully reproducible from
`(seed, schema_key, volume, variant)` without shipping the corpus itself. That matters at
volume=5000, where materializing documents onto the wire would be absurd.

Design constraints:

  - Noise must be *plausibly* relevant, not random filler. If irrelevant documents are
    obviously irrelevant, search solves the task and we are measuring retrieval.
  - Noise, decoys and signals all draw teams and product names from the same `Setting`. If
    only signal documents named the real teams, signals would be separable by vocabulary
    alone - a leak that would quietly make the benchmark trivial.
  - Prior-supporting decoys are the adversarial half: documents that genuinely support the
    stereotype answer, specific to this world. An agent reading surface features instead of
    evidence should be able to build a confident, wrong case from them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from insight_consolidation_v1.schemas import Setting

DOC_TYPES = ("email", "meeting_note", "ticket", "crm_note", "call_summary")

FIRST_NAMES = (
    "Marta", "Devan", "Priya", "Tomas", "Ingrid", "Rahul", "Noor", "Kaspar", "Lena",
    "Adaeze", "Yusuf", "Bea", "Hendrik", "Sofia", "Ravi", "Anneke", "Milos", "Chiara",
    "Oskar", "Naledi", "Jun", "Freya", "Emeka", "Dara", "Piotr",
)
LAST_NAMES = (
    "Okonkwo", "Lindqvist", "Baranov", "Ferreira", "Nakamura", "Duarte", "Haddad",
    "Kowalski", "Bergstrom", "Osei", "Vandermeer", "Rossi", "Novak", "Achebe", "Steiner",
)

# Account names are industry-matched to the schema's setting. Only the name varies between
# episodes - the world does not.
COMPANIES_BY_INDUSTRY: dict[str, tuple[str, ...]] = {
    "healthcare": ("Brightwater Health", "Calderon Health System", "Ridgemont Care"),
    "retail": ("Kestrel Retail Group", "Harlow & Vance", "Northfield Stores"),
    "services": ("Meridian Mechanical", "Ashgrove Facilities", "Talbot Services"),
    "insurance": ("Halbrook Mutual", "Sennett Insurance", "Aldermere Assurance"),
    "payments": ("Solvent Payments", "Kirkwall Financial", "Marrow Pay"),
    "distribution": ("Stanton Freight Foods", "Vellacourt Distribution", "Orrin Wholesale"),
    "manufacturing": ("Verrall Industrial", "Dunmore Manufacturing", "Ostrow Components"),
    "logistics": ("Northgate Logistics", "Pallister Freight", "Cairn Transport"),
    "fintech": ("Tallow Finance", "Berrick Money", "Sable Credit"),
}

# Ambient account activity. Plausible, on-topic, carrying nothing about the latent driver.
# {team} and {product} come from the schema's Setting, so noise is indistinguishable from
# signal by vocabulary.
NOISE_TEMPLATES = (
    "Confirmed the {team} session on the {ord} and moved it thirty minutes later at their request.",
    "{name} asked for the recording of the last {product_short} walkthrough to share with a colleague who missed it.",
    "Sent the updated onboarding checklist to {name} in {team}. No response needed.",
    "Routine check-in with {team}. Nothing outstanding, next touch in two weeks.",
    "{name} reported a display issue in the {product_short} export view. Reproduced, ticketed, fixed in the following release.",
    "Sent over the standard {product_short} implementation timeline and the list of prerequisites.",
    "Discussed holiday coverage for {team} over the summer period.",
    "{name} asked whether {product_short} single sign-on setup needs their identity team. Confirmed it does.",
    "Follow-up on an invoice address change requested by {company} accounts payable.",
    "Quick call to walk {name} through the {product_short} reporting view. Went fine, no follow-ups.",
    "{name} forwarded an industry newsletter item and asked if we had a view on it.",
    "Rescheduled the {team} workshop because of a conflicting all-hands at {company}.",
    "Access request for two new starters in {team}, processed same day.",
    "{name} noted the documentation link in the {product_short} welcome email is out of date.",
    "Agreed to send {team} a summary of the last three sessions for their internal notes.",
    "Housekeeping: updated the {company} account record with the new office address.",
    "{name} asked about the {product_short} release cadence out of general curiosity.",
    "Confirmed the {product_short} sandbox stays available to {team} through the evaluation.",
    "Short thread about scheduling across time zones for the {team} review.",
    "Passed {name} a customer story from a similar {team} organization.",
    "{name} asked whether {product_short} status notifications can go to a shared mailbox.",
    "Confirmed {team} headcount for the training session: nine, two dialling in.",
)

# Documents that genuinely support the stereotype prior, in this world.
PRIOR_TEMPLATES = {
    "price_sensitive": (
        "{name} asked for the annual {product_short} figure again and whether there is a multi-year discount.",
        "Procurement sent a benchmark of comparable tools with pricing highlighted.",
        "{name} mentioned budgets across the group were reviewed downward this year.",
        "Asked whether {team} could start on a reduced seat count and grow into it.",
    ),
    "technical_gaps": (
        "{name} listed three capabilities they could not find in the {product_short} trial.",
        "{team} asked for a written comparison against what they run today.",
        "{team} raised concerns about throughput at their expected volume.",
        "Asked for a second technical deep-dive before the next stage.",
    ),
    "strong_urgency": (
        "{name} asked twice this week about the earliest possible {product_short} start date.",
        "Pushed to compress the security review to fit before quarter close.",
        "{name} offered to chase {company} legal personally to speed things up.",
        "Asked whether onboarding for {team} could run in parallel with contracting.",
    ),
    "integration_risk": (
        "{name} asked for a diagram of how {product_short} sits alongside their warehouse.",
        "{team} asked about migration effort for existing records.",
        "Raised a concern about running two systems during any transition.",
        "Asked which {product_short} endpoints {team} would have to build against.",
    ),
    "value_driven": (
        "{name} asked for a template to model {team} time saved per week.",
        "Requested case studies with quantified outcomes for the board pack.",
        "Discussed how {company} would measure success at six and twelve months.",
        "Asked what the payback period usually looks like for a team the size of {team}.",
    ),
    "cautious_evaluator": (
        "{name} asked for a written pilot plan with explicit exit criteria.",
        "Requested references from two organizations of a similar size to {company}.",
        "Asked what a failed {product_short} evaluation usually looks like and why.",
        "Asked for the risk register from a comparable {team} deployment.",
    ),
    "slow_procurement": (
        "Legal returned the {product_short} agreement with standard markup, nothing unusual.",
        "{name} explained the three approval stages and their typical duration.",
        "Vendor onboarding form submitted; confirmation expected within two weeks.",
        "Procurement asked for our supplier registration details again.",
    ),
    "healthy_expansion": (
        "The {product_short} usage summary in the review showed month-on-month growth.",
        "{name} asked about pricing for additional seats next year.",
        "{team} was named internally as a success story at the operations all-hands.",
        "Asked whether {product_short} could be extended to a second site.",
    ),
    "satisfied_customer": (
        "{name} replied that everything is working well and nothing is outstanding.",
        "Support volume for {company} is at its lowest in twelve months.",
        "Renewal paperwork returned without comment or negotiation.",
        "{team} rated the last support interaction five out of five.",
    ),
    "workflow_gap": (
        "{name} walked us through the manual steps {team} repeats every week.",
        "{team} documented the handoff that currently breaks down.",
        "Discussed how much of the process is still spreadsheet-based.",
        "Asked how other {team} organizations handle the same gap.",
    ),
}

# Appended to a fraction of ALL document bodies - signal, decoy and noise alike. Two jobs:
# it lifts the number of distinct noise strings from a few hundred into the tens of
# thousands, and it keeps sentence structure uniform across categories. Without it, noise
# repeats verbatim at high volume while signals never do, and "text that appears exactly
# once" becomes a free shortlist of the entailing documents.
TAILS = (
    "Logged for the record.",
    "No action needed from our side.",
    "Copied {name} for visibility.",
    "Filed under the {company} account.",
    "Next touch scheduled.",
    "Nothing further raised on the call.",
    "Noted during the weekly {team} review.",
    "Follow-up left open pending their reply.",
    "Shared with the account team afterwards.",
    "Recorded at {name}'s request.",
    "Added to the running notes for {team}.",
    "No timeline attached.",
    "Confirmed by email the same day.",
    "Raised briefly, not pursued.",
    "Left with them to come back on.",
    "Mentioned in passing, not on the agenda.",
    "Picked up again the following week.",
    "Captured here for continuity.",
)

SUBJECT_BY_TYPE = {
    "email": ("Re: {company} - next steps", "Re: {company} / {product_short}", "{company} // follow-up"),
    "meeting_note": ("{company} - session notes", "Notes: {company} {team} review"),
    "ticket": ("[{company}] support request", "[{company}] question from {team}"),
    "crm_note": ("Account note - {company}", "Activity log - {company}"),
    "call_summary": ("Call summary - {company}", "Call notes - {company} {team}"),
}


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_type: str
    day: int
    subject: str
    author: str
    body: str

    def render(self, *, full: bool = True) -> str:
        head = f"[{self.doc_id}] ({self.doc_type}, day {self.day}) {self.subject} - {self.author}"
        return f"{head}\n{self.body}" if full else head

    def snippet(self, limit: int = 180) -> str:
        text = self.body if len(self.body) <= limit else self.body[: limit - 1] + "…"
        return f"[{self.doc_id}] ({self.doc_type}, day {self.day}) {text}"


class Personae:
    """The cast and account for one episode, drawn from the schema's fixed Setting.

    People and the account name vary between episodes. The product, the teams and the thing
    at stake do not - that is what stops the corpus inventing a different world each time.
    """

    def __init__(self, setting: Setting, rng: random.Random) -> None:
        self.setting = setting
        companies = COMPANIES_BY_INDUSTRY.get(setting.industry, ("Northgate Logistics",))
        self.company = rng.choice(companies)
        firsts = rng.sample(FIRST_NAMES, 6)
        lasts = rng.sample(LAST_NAMES, 6)
        self.people = [f"{f} {s}" for f, s in zip(firsts, lasts)]
        self.buyer = self.people[0]
        self.teams = setting.parties
        self.product_short = setting.vendor

    def body(self, text: str, rng: random.Random) -> str:
        """Realize a template into a document body, with a shared tail applied uniformly
        across signals, decoys and noise so category is not inferable from structure."""
        rendered = self.fill(text, rng)
        if rng.random() < 0.55:
            rendered = f"{rendered} {self.fill(rng.choice(TAILS), rng)}"
        return rendered

    def fill(self, text: str, rng: random.Random) -> str:
        return text.format(
            name=rng.choice(self.people),
            company=self.company,
            team=rng.choice(self.teams),
            product=self.setting.product,
            product_short=self.product_short,
            ord=rng.choice(("3rd", "9th", "14th", "21st", "27th")),
        )


def make_document(
    idx: int,
    doc_type: str,
    body: str,
    personae: Personae,
    rng: random.Random,
    day: int,
) -> Document:
    subject = personae.fill(rng.choice(SUBJECT_BY_TYPE[doc_type]), rng)
    return Document(
        doc_id=f"doc_{idx:05d}",
        doc_type=doc_type,
        day=day,
        subject=subject,
        author=rng.choice(personae.people),
        body=body,
    )
