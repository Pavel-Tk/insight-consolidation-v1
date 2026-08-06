"""Latent driver schemas - the domain layer, and the highest-leverage file in this repo.

Every episode starts from one `LatentSchema`. The schema is the ground truth: we sample it
first, then manufacture evidence that entails it only in aggregate. No single document ever
states the driver.

Each schema carries a fully specified `Setting`: what we sell, who the customer is, which
named internal teams exist, and what concrete thing is at stake. Nothing is left for a model
to invent. This matters twice over - it stops the corpus drifting into generic
consultant-speak, and it closes a leak. If signal documents named specific teams while noise
documents named random ones, signals would be separable by vocabulary alone and the task
would collapse into string matching.

A schema is a claim about why a buyer actually behaves the way they do, plus:

  - `stated_surface`  what the buyer says instead (the decoy objection they voice out loud)
  - `prior_key`       the stereotype an agent lands on if it reads surface features instead
                      of evidence. This is the anti-prior trap.
  - `signals`         atomic behavioural traces. Each becomes ONE document fragment.
                      Individually ambiguous; jointly they entail the driver.
  - `revises_to`      an alternative driver that late evidence can flip the episode to.

EDIT THIS FILE. Rules of thumb:

  1. No signal may name the driver. If a signal lets a careful reader answer from that one
     document, it is too strong - weaken it or split it. Enforced by
     `test_no_single_document_names_the_driver`.
  2. At least two signals must be behavioural (what they did) rather than verbal (what they
     said). Buyers misreport their own motives; that is the whole premise.
  3. The `prior_key` must be genuinely plausible. If the stereotype is obviously wrong, the
     anti-prior split measures nothing.
  4. Be specific. Name the system, the team, the release, the amount, the form. Vagueness is
     where both hallucination and unearned difficulty come from.
  5. Signals should be separated in time and channel. Consolidation across sources is the
     skill under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DocType = str  # "email" | "meeting_note" | "ticket" | "crm_note" | "call_summary"


@dataclass(frozen=True)
class Setting:
    """The fixed world an episode happens in. Fully named, so nothing is invented."""

    vendor: str
    """Us. The company doing the selling."""

    product: str
    """What we sell, named, with enough of a description to anchor the corpus."""

    industry: str
    """Key into `corpus.COMPANIES_BY_INDUSTRY`. Only the account name varies per episode."""

    parties: tuple[str, ...]
    """Named internal teams and roles in the customer org. Signals, decoys and ambient
    noise all draw from this list, so no team name is unique to signal documents."""

    contested: str = ""
    """The specific thing at stake, where the schema turns on one."""


@dataclass(frozen=True)
class Signal:
    """One atomic, individually-insufficient trace of the latent driver."""

    text: str
    """Realized into a document body. `{name}` is a person, `{company}` the account."""

    doc_type: DocType = "email"

    behavioural: bool = False
    """True if this is something the buyer DID rather than SAID. See rule 2."""

    weight: float = 1.0
    """Signals at weight >= 1.0 form the minimal entailing set - the citation answer key."""


@dataclass(frozen=True)
class LatentSchema:
    key: str
    label: str
    description: str
    """What the judge compares the agent's answer against. Write it as the sentence a very
    good account executive would say after three months on the account."""

    setting: Setting
    stated_surface: str
    prior_key: str
    prior_label: str
    signals: tuple[Signal, ...]
    revises_to: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def minimal_set(self) -> tuple[Signal, ...]:
        return tuple(s for s in self.signals if s.weight >= 1.0)


SCHEMAS: tuple[LatentSchema, ...] = (
    LatentSchema(
        key="career_exposure",
        label="Personal blame risk, not budget",
        setting=Setting(
            vendor="Ledgerline",
            product="Ledgerline, an accounts-payable automation platform",
            industry="healthcare",
            parties=(
                "Finance Shared Services",
                "IT Applications",
                "Internal Audit",
                "Procurement",
            ),
            contested="replacing the manual three-way match in Finance Shared Services",
        ),
        description=(
            "The VP of Finance Operations is not price sensitive. They are afraid of being "
            "personally blamed if this goes the way the 2023 Workday Financials rollout "
            "did, so they are buying cover rather than capability - more names on the "
            "decision, more documentation, more people who cannot later say they were not "
            "consulted. Price is the socially acceptable version of that fear."
        ),
        stated_surface="the price is hard to justify to the finance committee",
        prior_key="price_sensitive",
        prior_label="A price-driven buyer who needs a discount to sign",
        signals=(
            Signal(
                "Asked twice whether the VP Finance at Brightwater would go on record as a "
                "reference in the committee paper, then never took the intro when it was "
                "offered.",
                "crm_note",
                behavioural=True,
            ),
            Signal(
                "Forwarded the Ledgerline SOC 2 report to Internal Audit and two names from "
                "IT Applications who had not been on any prior thread. No message attached.",
                "email",
                behavioural=True,
            ),
            Signal(
                "Mentioned the 2023 Workday Financials rollout in passing - said it 'did not "
                "end well for the person who owned it'.",
                "call_summary",
            ),
            Signal(
                "Moved the pricing call twice, and personally attended all four three-way "
                "match walkthroughs with Finance Shared Services.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Asked whether any Ledgerline customer had ever reverted to manual matching, "
                "and what happened to the relationship afterwards.",
                "call_summary",
                weight=0.5,
            ),
        ),
        revises_to="budget_freeze",
        tags=("enterprise", "economic_buyer"),
    ),
    LatentSchema(
        key="internal_political_stalemate",
        label="Two teams are fighting and we are the proxy",
        setting=Setting(
            vendor="Halyard",
            product="Halyard, a customer data platform",
            industry="retail",
            parties=(
                "Marketing Technology",
                "Data Platform Engineering",
                "Digital Analytics",
                "Enterprise Architecture",
            ),
            contested="who owns customer identity resolution",
        ),
        description=(
            "The evaluation is not about Halyard. Marketing Technology wants identity "
            "resolution to live in a bought CDP they control; Data Platform Engineering "
            "wants it in the Snowflake warehouse they already run. The purchase decides that "
            "ownership question, so nothing closes until it is settled - however well the "
            "evaluation goes."
        ),
        stated_surface="we are still working through the technical evaluation",
        prior_key="technical_gaps",
        prior_label="A technically unconvinced buyer who needs more proof of capability",
        signals=(
            Signal(
                "Marketing Technology and Data Platform Engineering each ran a Halyard trial "
                "without telling the other. Both asked us not to mention it to the other.",
                "crm_note",
                behavioural=True,
            ),
            Signal(
                "Identity resolution scored 4.6/5 on their own scorecard. The next meeting "
                "was still about which team would own the match rules.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Asked whether the Halyard contract could be split across the Marketing "
                "Technology and Data Platform Engineering cost centres.",
                "email",
            ),
            Signal(
                "The Enterprise Architecture director joined the last call, asked nothing, "
                "and left after eleven minutes.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Someone in Digital Analytics referred to the project as 'the Data Platform "
                "team's idea'.",
                "call_summary",
                weight=0.5,
            ),
        ),
        tags=("enterprise", "multi_threaded"),
    ),
    LatentSchema(
        key="champion_is_leaving",
        label="Our champion is on the way out",
        setting=Setting(
            vendor="Fieldpath",
            product="Fieldpath, a field service scheduling and dispatch platform",
            industry="services",
            parties=(
                "Service Operations",
                "Dispatch",
                "IT",
                "HR Operations",
            ),
            contested="the dispatch analytics workstream",
        ),
        description=(
            "The Director of Service Operations driving this is quietly on their way out of "
            "the company. Their enthusiasm is real and their remaining influence is not. The "
            "urgency is personal - they want it live before they go - and the deal has no "
            "sponsor once they leave."
        ),
        stated_surface="I want Fieldpath live before the quarter ends",
        prior_key="strong_urgency",
        prior_label="A highly motivated champion driving a fast close",
        signals=(
            Signal(
                "Stopped copying Dispatch on the rollout thread. They had been on every "
                "message for four months.",
                "email",
                behavioural=True,
            ),
            Signal(
                "Pushed hard for a go-live date and was vague about who runs dispatch "
                "configuration after onboarding.",
                "call_summary",
            ),
            Signal(
                "Declined to introduce us to whoever picks up the dispatch analytics "
                "workstream, saying it is 'in flux'.",
                "crm_note",
                behavioural=True,
            ),
            Signal(
                "Changed their contact address on the shared implementation plan to a "
                "personal one.",
                "ticket",
                behavioural=True,
            ),
            Signal(
                "Asked whether Fieldpath configuration decisions would be documented for "
                "people joining Service Operations later.",
                "meeting_note",
                weight=0.5,
            ),
        ),
        tags=("mid_market", "champion"),
    ),
    LatentSchema(
        key="already_built_internally",
        label="They have a home-grown version they are protecting",
        setting=Setting(
            vendor="Cormorant",
            product="Cormorant, an intelligent document processing platform",
            industry="insurance",
            parties=(
                "Claims Engineering",
                "Claims Operations",
                "Enterprise Architecture",
                "Procurement",
            ),
            contested="whether Project Tern gets replaced or extended",
        ),
        description=(
            "Claims Engineering built Project Tern two years ago - an in-house OCR and rules "
            "pipeline that handles roughly 60% of inbound ACORD forms. It cannot read "
            "handwritten loss-run supplements, which is the only part they want to buy. The "
            "evaluation is partly an exercise in proving Tern was the right call, and buying "
            "the whole platform reads to that team as an admission it was not."
        ),
        stated_surface="we need to check how Cormorant fits our existing stack",
        prior_key="integration_risk",
        prior_label="A buyer worried about integration complexity",
        signals=(
            Signal(
                "Claims Engineering asked for the Cormorant API reference before anyone "
                "asked for a demo.",
                "email",
                behavioural=True,
            ),
            Signal(
                "An engineer asked twice how Cormorant handles multi-page ACORD 125 "
                "attachments where page two is rotated - a question you only reach after "
                "hitting it yourself.",
                "ticket",
            ),
            Signal(
                "Asked for pricing on handwritten loss-run supplements only, described as "
                "'the part Tern never got to'.",
                "crm_note",
            ),
            Signal(
                "Their evaluation criteria document uses field names from Tern's schema that "
                "appear nowhere in Cormorant's materials.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Project Tern came up by name three times in one call with no explanation of "
                "what it is.",
                "call_summary",
                weight=0.5,
            ),
        ),
        tags=("enterprise", "technical_buyer"),
    ),
    LatentSchema(
        key="compliance_deadline_not_value",
        label="A dated obligation is driving this, not the value case",
        setting=Setting(
            vendor="Sentry Trace",
            product="Sentry Trace, an access review and entitlement certification platform",
            industry="payments",
            parties=(
                "Security Governance",
                "Legal",
                "Platform Engineering",
                "Internal Audit",
            ),
            contested="quarterly access reviews before the PCI DSS 4.0 assessment",
        ),
        description=(
            "The driver is the PCI DSS 4.0 assessment in the second week of November and the "
            "quarterly access review evidence they do not currently produce. They will buy "
            "something before that date more or less regardless of fit. The value "
            "conversation is theatre, and the real competition is the cheapest thing that "
            "satisfies the assessor."
        ),
        stated_surface="we are excited about the productivity gains for Platform Engineering",
        prior_key="value_driven",
        prior_label="A buyer motivated by ROI and engineering productivity",
        signals=(
            Signal(
                "Asked what the fastest possible Sentry Trace deployment looks like if scope "
                "is cut to entitlement certification only.",
                "call_summary",
            ),
            Signal(
                "Legal was in the second meeting. Platform Engineering, who would actually "
                "run it, joined in the sixth.",
                "crm_note",
                behavioural=True,
            ),
            Signal(
                "Every date discussed - security review, procurement, go-live - anchors "
                "backwards from the second week of November.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Asked for written confirmation of specific control language, pasted in with "
                "another vendor's formatting still on it.",
                "email",
                behavioural=True,
            ),
            Signal(
                "Declined the Sentry Trace roadmap session twice.",
                "meeting_note",
                weight=0.5,
            ),
        ),
        tags=("enterprise", "regulated"),
    ),
    LatentSchema(
        key="pilot_as_free_consulting",
        label="The pilot is the deliverable, not a step toward one",
        setting=Setting(
            vendor="Northlight",
            product="Northlight, a supply chain network optimization platform",
            industry="distribution",
            parties=(
                "Supply Chain Planning",
                "Distribution Operations",
                "Finance",
                "Procurement",
            ),
            contested="the depot consolidation analysis the pilot produces",
        ),
        description=(
            "They want the depot consolidation map and network flow analysis that come out "
            "of the Northlight pilot, and they have no intention of buying the platform. "
            "Supply Chain Planning has a consolidation decision to make and no analyst "
            "capacity to make it. Once they have the output, this will go quiet."
        ),
        stated_surface="if the pilot goes well we will move to a full rollout",
        prior_key="cautious_evaluator",
        prior_label="A methodical buyer de-risking before committing",
        signals=(
            Signal(
                "Negotiated the pilot's analysis scope line by line and accepted the pilot "
                "fee without a single question.",
                "crm_note",
                behavioural=True,
            ),
            Signal(
                "Asked for the depot consolidation output as an editable file without "
                "Northlight branding.",
                "email",
                behavioural=True,
            ),
            Signal(
                "Procurement has not been involved at any point in eleven weeks.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Sent three years of shipment data within a day of us asking. Every other "
                "request in the cycle has taken a fortnight.",
                "ticket",
                behavioural=True,
            ),
            Signal(
                "Deflected the question of what happens after the pilot twice in the same "
                "call.",
                "call_summary",
                weight=0.5,
            ),
        ),
        tags=("mid_market", "pilot"),
    ),
    LatentSchema(
        key="budget_freeze",
        label="The money is gone and nobody has said so",
        setting=Setting(
            vendor="Cadence",
            product="Cadence, a workforce planning platform",
            industry="manufacturing",
            parties=(
                "HR Operations",
                "Finance Business Partnering",
                "IT",
                "Group Procurement",
            ),
            contested="the fiscal-year start date for the Cadence licence",
        ),
        description=(
            "Group Finance froze this budget after the Q2 segment write-down, and the HR "
            "Operations lead has not been authorized to tell us. Activity continues because "
            "activity is free. Anything that requires a commitment does not move, and the "
            "pattern is visible in which meetings get rescheduled."
        ),
        stated_surface="we are working through the approval process with Group Procurement",
        prior_key="slow_procurement",
        prior_label="A slow but healthy procurement cycle",
        signals=(
            Signal(
                "Technical sessions with IT are still easy to book. The commercial call has "
                "moved three times in five weeks.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Asked whether the Cadence licence could start in the next fiscal year at "
                "the same price.",
                "email",
            ),
            Signal(
                "Went quiet for eleven days, then replied to an unrelated HR Operations "
                "thread the same afternoon.",
                "crm_note",
                behavioural=True,
            ),
            Signal(
                "Asked for our smallest possible first-year commitment, framed as 'just so I "
                "have the number to hand'.",
                "call_summary",
            ),
            Signal(
                "A hiring freeze in Finance Business Partnering came up in an unrelated "
                "scheduling exchange.",
                "ticket",
                weight=0.5,
            ),
        ),
        tags=("enterprise", "commercial"),
    ),
    LatentSchema(
        key="usage_concentrated_in_one_team",
        label="Adoption is one team deep and looks broader than it is",
        setting=Setting(
            vendor="Beacon",
            product="Beacon, an incident management platform",
            industry="logistics",
            parties=(
                "Network Operations Centre",
                "IT Service Management",
                "Application Support",
                "Procurement",
            ),
            contested="whether the renewal expands beyond the NOC night shift",
        ),
        description=(
            "Beacon adoption is real and it is almost entirely the Network Operations Centre "
            "night shift, whose escalation workflow is unusual. The account looks healthy in "
            "aggregate and is fragile underneath: if the NOC shift lead moves, usage "
            "collapses. Expansion is being discussed with people in IT Service Management "
            "who have never used the product."
        ),
        stated_surface="adoption is going well across the operations org",
        prior_key="healthy_expansion",
        prior_label="A healthy account ready for expansion",
        signals=(
            Signal(
                "The same four Network Operations Centre names appear on every Beacon ticket "
                "for six months.",
                "ticket",
                behavioural=True,
            ),
            Signal(
                "The IT Service Management director asked in the QBR how escalation routing "
                "is configured. Anyone using Beacon weekly would know.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Every feature request this year describes the same night-shift handover "
                "workflow.",
                "ticket",
                behavioural=True,
            ),
            Signal(
                "The named renewal contact in IT Service Management has never logged in.",
                "crm_note",
                behavioural=True,
            ),
            Signal(
                "Beacon training offered to Application Support was accepted, then nobody "
                "attended.",
                "email",
                weight=0.5,
            ),
        ),
        tags=("customer_success", "renewal"),
    ),
    LatentSchema(
        key="trust_damaged_earlier",
        label="Something went wrong before and nobody raised it",
        setting=Setting(
            vendor="Ledgerline",
            product="Ledgerline, an accounts-payable automation platform",
            industry="healthcare",
            parties=(
                "Finance Shared Services",
                "Treasury",
                "IT Applications",
                "Internal Audit",
            ),
            contested="whether the relationship survives to renewal",
        ),
        description=(
            "The Ledgerline 4.2 release duplicated about $180,000 of vendor payments and it "
            "took us six weeks to credit them back. We never formally acknowledged it. "
            "Finance Shared Services has been polite and transactional ever since, and that "
            "politeness is being mistaken for health. Nothing expands until the original "
            "incident is named."
        ),
        stated_surface="everything is fine, no concerns from our side",
        prior_key="satisfied_customer",
        prior_label="A satisfied, low-maintenance customer",
        signals=(
            Signal(
                "Stopped attending the monthly Finance Shared Services sync in March. Never "
                "cancelled the invite.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Replies are three lines where they used to be three paragraphs, and contain "
                "no questions.",
                "email",
                behavioural=True,
            ),
            Signal(
                "Routed a payment-run question through the support queue. They used to send "
                "those straight to their account manager.",
                "ticket",
                behavioural=True,
            ),
            Signal(
                "Declined to be a reference for the healthcare case study, citing policy. "
                "They had agreed to one in 2024.",
                "crm_note",
                behavioural=True,
            ),
            Signal(
                "Referred to the period around the 4.2 release as 'after all that'.",
                "call_summary",
                weight=0.5,
            ),
        ),
        tags=("customer_success", "churn_risk"),
    ),
    LatentSchema(
        key="buying_for_a_person_not_a_problem",
        label="This is being bought to keep one person happy",
        setting=Setting(
            vendor="Tessera",
            product="Tessera, an experimentation and feature flagging platform",
            industry="fintech",
            parties=(
                "Product",
                "Growth Engineering",
                "Data Science",
                "Procurement",
            ),
            contested="whether Tessera survives the new VP Product's interest",
        ),
        description=(
            "Tessera is being bought to retain the VP Product who joined four months ago and "
            "asked for it by name, having used it at their previous employer. The stated "
            "experimentation gap is real but secondary, and the evaluation criteria track "
            "that one person's preferences. If they lose interest or leave, the rationale "
            "goes with them."
        ),
        stated_surface="this closes a gap in how we run experiments",
        prior_key="workflow_gap",
        prior_label="A buyer solving a genuine documented workflow gap",
        signals=(
            Signal(
                "The requirements document was rewritten after the VP Product's comments and "
                "not after Data Science's.",
                "meeting_note",
                behavioural=True,
            ),
            Signal(
                "Requests cite the VP Product as the reason rather than as a stakeholder - "
                "'this is what she wants it to do'.",
                "email",
            ),
            Signal(
                "Nobody in Growth Engineering has been able to say what happens if Tessera "
                "is not bought.",
                "call_summary",
                behavioural=True,
            ),
            Signal(
                "The evaluation criteria include a mutually-exclusive holdout model that "
                "matches one vendor's implementation and no other.",
                "crm_note",
                behavioural=True,
            ),
            Signal(
                "The VP Product joined four months ago and chairs the review despite not "
                "owning the budget.",
                "ticket",
                weight=0.5,
            ),
        ),
        tags=("startup", "political"),
    ),
)

SCHEMAS_BY_KEY = {s.key: s for s in SCHEMAS}


def get_schema(key: str) -> LatentSchema:
    return SCHEMAS_BY_KEY[key]
