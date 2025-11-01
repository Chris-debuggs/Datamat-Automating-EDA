"use client"

import { useEffect, useRef } from "react"

interface AnimatedRobotProps {
  className?: string
  size?: "sm" | "md" | "lg"
}

export function AnimatedRobot({ className = "", size = "lg" }: AnimatedRobotProps) {
  const robotRef = useRef<HTMLDivElement>(null)

  const sizeClasses = {
    sm: "w-20 h-28",
    md: "w-32 h-44",
    lg: "w-40 h-56",
  }

  const handleRobotClick = () => {
    if (!robotRef.current) return

    const robot = robotRef.current
    const mainEye = robot.querySelector(".main-eye") as HTMLElement
    const indicators = robot.querySelectorAll(".indicator") as NodeListOf<HTMLElement>
    const antennas = robot.querySelectorAll(".antenna") as NodeListOf<HTMLElement>

    // Main eye reaction - intense glow
    if (mainEye) {
      mainEye.style.boxShadow = "0 0 40px rgba(255, 46, 99, 1), inset 0 2px 10px rgba(0,0,0,0.8)"
      mainEye.style.transform = "translateX(-50%) scale(1.1)"
      setTimeout(() => {
        mainEye.style.boxShadow = "0 0 20px rgba(255, 46, 99, 0.3), inset 0 2px 10px rgba(0,0,0,0.8)"
        mainEye.style.transform = "translateX(-50%) scale(1)"
      }, 400)
    }

    // All indicators flash simultaneously
    indicators.forEach((indicator) => {
      indicator.style.background = "#ffffff"
      indicator.style.boxShadow = "0 0 10px rgba(255, 255, 255, 0.8)"
      setTimeout(() => {
        indicator.style.background = "#FF2E63"
        indicator.style.boxShadow = "none"
      }, 200)
    })

    // Antennas vibrate
    antennas.forEach((antenna, index) => {
      const isLeft = index === 0
      antenna.style.transform = `rotate(${isLeft ? -20 : 20}deg)`
      setTimeout(() => {
        antenna.style.transform = `rotate(${isLeft ? -12 : 12}deg)`
      }, 200)
    })

    // Robot enhanced bounce
    robot.style.transform = "translateY(-20px) rotate(2deg) scale(1.05)"
    setTimeout(() => {
      robot.style.transform = "translateY(0) rotate(0deg) scale(1)"
    }, 300)
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!robotRef.current) return

      const robotBody = robotRef.current.querySelector(".robot-body") as HTMLElement
      const mainEye = robotRef.current.querySelector(".main-eye") as HTMLElement

      if (!robotBody || !mainEye) return

      const bodyRect = robotBody.getBoundingClientRect()
      const centerX = bodyRect.left + bodyRect.width / 2
      const centerY = bodyRect.top + bodyRect.height / 2

      const angle = Math.atan2(e.clientY - centerY, e.clientX - centerX)
      const distance = Math.min(3, Math.sqrt(Math.pow(e.clientX - centerX, 2) + Math.pow(e.clientY - centerY, 2)) / 100)

      const moveX = Math.cos(angle) * distance
      const moveY = Math.sin(angle) * distance

      mainEye.style.transform = `translateX(-50%) translate(${moveX}px, ${moveY}px)`
    }

    document.addEventListener("mousemove", handleMouseMove)
    return () => document.removeEventListener("mousemove", handleMouseMove)
  }, [])

  return (
    <div className={`${sizeClasses[size]} ${className} relative`}>
      <style jsx>{`
        .robot {
          width: 100%;
          height: 100%;
          margin: 0 auto;
          position: relative;
          cursor: pointer;
          animation: gentle-float 4s ease-in-out infinite;
          filter: drop-shadow(0 10px 20px rgba(0,0,0,0.3));
          transition: all 0.3s ease;
        }

        .robot-body {
          width: 70%;
          height: 64%;
          background: linear-gradient(145deg, #f5f5f5, #e0e0e0);
          border: 3px solid #000000;
          border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
          position: absolute;
          top: 14%;
          left: 15%;
          transition: all 0.3s ease;
          overflow: hidden;
          box-shadow: inset 0 2px 10px rgba(0,0,0,0.1), 4px 4px 0px 0px rgba(0,0,0,1);
        }

        .robot:hover .robot-body {
          transform: translateY(-5px);
        }

        .orange-stripe {
          width: 20%;
          height: 100%;
          background: linear-gradient(180deg, #FF2E63, #e55a2b);
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          top: 0;
        }

        .texture-overlay {
          position: absolute;
          width: 100%;
          height: 100%;
          background: 
            radial-gradient(circle at 30% 20%, rgba(0,0,0,0.05) 2px, transparent 2px),
            radial-gradient(circle at 70% 40%, rgba(0,0,0,0.03) 1px, transparent 1px),
            radial-gradient(circle at 20% 80%, rgba(0,0,0,0.04) 1.5px, transparent 1.5px);
          background-size: 20px 20px, 15px 15px, 25px 25px;
        }

        .antenna {
          width: 3px;
          height: 22%;
          background: linear-gradient(to top, #333, #666);
          position: absolute;
          top: 4%;
          border-radius: 2px;
          transition: all 0.2s ease;
        }

        .antenna.left {
          left: 22%;
          transform: rotate(-12deg);
        }

        .antenna.right {
          right: 22%;
          transform: rotate(12deg);
        }

        .antenna::after {
          content: '';
          width: 8px;
          height: 8px;
          background: radial-gradient(circle, #FF2E63, #e55a2b);
          border: 1px solid #000000;
          border-radius: 50%;
          position: absolute;
          top: -6px;
          left: 50%;
          transform: translateX(-50%);
          animation: antenna-pulse 2s infinite;
          box-shadow: 0 0 8px rgba(255, 46, 99, 0.6);
        }

        .main-eye {
          width: 31%;
          height: 18%;
          background: linear-gradient(145deg, #2a2a2a, #000000);
          border: 3px solid #333333;
          border-radius: 50%;
          position: absolute;
          top: 25%;
          left: 50%;
          transform: translateX(-50%);
          transition: all 0.2s ease;
          box-shadow: 
            0 0 20px rgba(255, 46, 99, 0.3),
            inset 0 2px 10px rgba(0,0,0,0.8);
        }

        .main-eye::before {
          content: '';
          width: 60%;
          height: 60%;
          background: radial-gradient(circle at 35% 35%, #FF2E63, #e55a2b, #000000);
          border-radius: 50%;
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          animation: eye-glow 3s infinite;
        }

        .main-eye::after {
          content: '';
          width: 20%;
          height: 20%;
          background: #ffffff;
          border-radius: 50%;
          position: absolute;
          top: 30%;
          left: 40%;
          opacity: 0.8;
        }

        .side-sensor {
          width: 12%;
          height: 5%;
          background: linear-gradient(145deg, #333333, #1a1a1a);
          border: 1px solid #000000;
          border-radius: 3px;
          position: absolute;
          top: 29%;
          box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
        }

        .side-sensor.left {
          left: -6%;
        }

        .side-sensor.right {
          right: -6%;
        }

        .indicator {
          width: 6px;
          height: 6px;
          background: #FF2E63;
          border: 1px solid #000000;
          border-radius: 50%;
          position: absolute;
          animation: blink 3s infinite;
        }

        .indicator.top {
          top: 18%;
          left: 50%;
          transform: translateX(-50%);
          animation-delay: 0s;
        }

        .indicator.bottom-left {
          bottom: 14%;
          left: 22%;
          animation-delay: 1s;
        }

        .indicator.bottom-right {
          bottom: 14%;
          right: 22%;
          animation-delay: 2s;
        }

        .arm {
          position: absolute;
          top: 43%;
        }

        .arm.left {
          left: 3%;
        }

        .arm.right {
          right: 3%;
        }

        .arm-segment {
          width: 9%;
          height: 11%;
          background: linear-gradient(145deg, #FF2E63, #e55a2b);
          border: 2px solid #000000;
          border-radius: 8px;
          position: relative;
          box-shadow: inset 0 1px 3px rgba(255,255,255,0.3);
        }

        .arm-joint {
          width: 8px;
          height: 8px;
          background: #000000;
          border: 1px solid #333333;
          border-radius: 50%;
          position: absolute;
          bottom: -4px;
          left: 50%;
          transform: translateX(-50%);
        }

        .hand {
          width: 75%;
          height: 71%;
          background: linear-gradient(145deg, #2a2a2a, #000000);
          border: 1px solid #000000;
          border-radius: 6px;
          position: absolute;
          bottom: -89%;
          left: 50%;
          transform: translateX(-50%);
          animation: hand-wiggle 4s infinite;
        }

        .arm.left .hand {
          animation-delay: 0s;
        }

        .arm.right .hand {
          animation-delay: 2s;
        }

        .body-base {
          width: 38%;
          height: 7%;
          background: linear-gradient(145deg, #333333, #1a1a1a);
          border: 2px solid #000000;
          border-radius: 10px;
          position: absolute;
          bottom: 4%;
          left: 50%;
          transform: translateX(-50%);
          box-shadow: 0 5px 10px rgba(0,0,0,0.3);
        }

        @keyframes gentle-float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-12px) rotate(1deg); }
        }

        @keyframes antenna-pulse {
          0%, 100% { 
            opacity: 1; 
            box-shadow: 0 0 8px rgba(255, 46, 99, 0.6);
          }
          50% { 
            opacity: 0.7; 
            box-shadow: 0 0 15px rgba(255, 46, 99, 1);
          }
        }

        @keyframes eye-glow {
          0%, 100% { 
            box-shadow: 0 0 15px rgba(255, 46, 99, 0.8);
          }
          50% { 
            box-shadow: 0 0 25px rgba(255, 46, 99, 1);
          }
        }

        @keyframes blink {
          0%, 90%, 100% { opacity: 1; }
          95% { opacity: 0.3; }
        }

        @keyframes hand-wiggle {
          0%, 100% { transform: translateX(-50%) rotate(0deg); }
          25% { transform: translateX(-50%) rotate(5deg); }
          75% { transform: translateX(-50%) rotate(-5deg); }
        }
      `}</style>

      <div ref={robotRef} className="robot" onClick={handleRobotClick}>
        {/* Main egg-shaped body */}
        <div className="robot-body">
          <div className="orange-stripe"></div>
          <div className="texture-overlay"></div>

          {/* Antennas */}
          <div className="antenna left"></div>
          <div className="antenna right"></div>

          {/* Main eye/camera */}
          <div className="main-eye"></div>

          {/* Side sensors */}
          <div className="side-sensor left"></div>
          <div className="side-sensor right"></div>

          {/* Indicator lights */}
          <div className="indicator top"></div>
          <div className="indicator bottom-left"></div>
          <div className="indicator bottom-right"></div>
        </div>

        {/* Mechanical arms */}
        <div className="arm left">
          <div className="arm-segment">
            <div className="arm-joint"></div>
          </div>
          <div className="hand"></div>
        </div>
        <div className="arm right">
          <div className="arm-segment">
            <div className="arm-joint"></div>
          </div>
          <div className="hand"></div>
        </div>

        {/* Body base */}
        <div className="body-base"></div>
      </div>
    </div>
  )
}
