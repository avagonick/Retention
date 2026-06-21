import FuzzyText from "./FuzzyText";

export default function HeroPage({ onStart }) {
  return (
    <div className="hero">
      <FuzzyText
        fontSize="clamp(3rem, 17vw, 13rem)"
        fontWeight={900}
        color="#211c14"
        baseIntensity={0.09}
        hoverIntensity={0.21}
        enableHover={true}
      >
        RETENTION
      </FuzzyText>
      <p className="hero-blurb">
        We use a brain-encoding model to optimize educational videos for how the
        brain actually remembers.
      </p>
      <button className="btn-get" onClick={onStart}>
        Get started
      </button>
    </div>
  );
}
