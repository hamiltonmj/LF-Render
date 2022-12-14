#pragma once

#include <GLFW/glfw3.h>
#include <map>

/// <summary>
/// Simple data structre used to describe generate a mapping system between actions and their corresponding command.
/// Within the lightfieldviewer these command are then described
/// </summary>
class ctrlMap
{
	std::map<std::pair<int,bool>, int> m_controlMap;

	std::pair<std::pair<int, bool>, int> mapping(int a, int b, int pressed = 1)
	{
		std::pair<int, int> x = std::pair<int, int>(a, pressed);
		return std::pair<std::pair<int, int>, int>(x, b);
	}
	
public:
	enum ctrls { forward, back, up, down, left, right, rotAround, turnCamera, load, setPos, fastCamOn, fastCamOff, exit, scroll, endTracking, hideHud};

	ctrlMap()
	{
		defaultMapping();
	}

	/// <summary>
	/// Resets the mapping of the ctrlmap to default keys, will not remove extra keys though 
	/// </summary>
	void ctrlMap::defaultMapping()
	{
		m_controlMap.insert(mapping(GLFW_KEY_ESCAPE, exit));
		m_controlMap.insert(mapping(GLFW_KEY_W, forward));
		m_controlMap.insert(mapping(GLFW_KEY_S, back));
		m_controlMap.insert(mapping(GLFW_KEY_A, left));
		m_controlMap.insert(mapping(GLFW_KEY_D, right));
		m_controlMap.insert(mapping(GLFW_KEY_E, up));
		m_controlMap.insert(mapping(GLFW_KEY_Q, down));
		m_controlMap.insert(mapping(GLFW_KEY_LEFT_SHIFT, fastCamOn));
		m_controlMap.insert(mapping(GLFW_KEY_LEFT_SHIFT, fastCamOff, 0));

		m_controlMap.insert(mapping(GLFW_KEY_ESCAPE, exit));
		m_controlMap.insert(mapping(GLFW_KEY_Q, down));
		m_controlMap.insert(mapping(GLFW_MOUSE_BUTTON_LEFT, rotAround));
		m_controlMap.insert(mapping(GLFW_MOUSE_BUTTON_RIGHT, turnCamera));
		m_controlMap.insert(mapping(-5, scroll));
		m_controlMap.insert(mapping(GLFW_MOUSE_BUTTON_LEFT, endTracking, 0));
		m_controlMap.insert(mapping(GLFW_MOUSE_BUTTON_RIGHT, endTracking, 0));
		m_controlMap.insert(mapping(-6, load, 1));
		m_controlMap.insert(mapping(GLFW_KEY_H, hideHud));
	}

	/// <summary>
	/// Given a action, return the corresponding action enum
	/// </summary>
	/// <param name="ctrl"> What control was hit and how ie "e" key and was pressed down</param>
	/// <returns> Returns the corresponding enum from ctrls so for example would return ctrls::up when given "e" key pressed down</returns>
	int ctrlMap::getctrl(std::pair<int, int> ctrl)
	{
		auto x = m_controlMap.find(ctrl);
		return (x == m_controlMap.end()) ? -1 : x->second;
	}

	/// <summary>
	/// Given a action and ctrl will add it to the ctrl map
	/// </summary>
	/// <param name="key"> Given key to map ie "e"</param>
	/// <param name="ctrl"> the command we want it to do ie: move up</param>
	/// <param name="action"> On what action do we want it to respond ie: keyborad press down etc</param>
	void ctrlMap::addctrl(int key, int ctrl, int action = 1)
	{
		m_controlMap.insert(mapping(key, ctrl, action));
	}
};


